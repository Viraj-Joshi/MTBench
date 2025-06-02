from typing import List, Tuple

import numpy as np
import time
import os

import torch 
from torch import nn
from torch import optim
import torch.distributed as dist
from torch.nn import functional as F
from scipy.optimize import minimize

from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common import datasets

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

def actor_loss_mt(old_action_neglog_probs_batch, action_neglog_probs, advantage, is_ppo, curr_e_clip, task_indices):
    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
    else:
        a_loss = (action_neglog_probs * advantage)

    return a_loss


class MTGRPOAgent(A2CAgent):
    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        self._device = self.config.get('device', 'cuda:0')
        self.shuffle_data = self.config.get('shuffle_data', False)

        self.all_task_indices : torch.Tensor = self.vec_env.env.extras["task_indices"]
        ordered_task_names : list[str] = self.vec_env.env.extras["ordered_task_names"]
        
        # arguments for building the network
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
            'task_indices': self.all_task_indices,
            'task_embedding_dim': torch.unique(self.all_task_indices).shape[0],
            'ordered_task_names': ordered_task_names,
            'device': self._device
        }
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)

        self.algo_observer.after_init(self)
        self.actor_loss_func = actor_loss_mt
        

    def inner_play_steps(self):
        self.update_list = ['actions', 'neglogpacs', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs) # calls the model to sample an action (is_train=False)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in self.update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            # env_done_indices = env_done_indices[env_done_indices < 512]
     
            # self.game_rewards.update(self.current_rewards[:512][env_done_indices])
            # self.game_shaped_rewards.update(self.current_shaped_rewards[:512][env_done_indices])
            # self.game_lengths.update(self.current_lengths[:512][env_done_indices])
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones


        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        # (episode_length, num_envs, 1)

        # uncomment below to use episodic return
        # mb_returns = mb_rewards.transpose(0, 1).sum(dim=1).squeeze(-1)
        
        ### Calculate reward-to-go ###
        mb_rewards_to_go = torch.zeros_like(mb_rewards)
        g_t_plus_1 = torch.zeros_like(mb_rewards)[0] # hold G_{t+1} as we iterate backwards.
        for t in reversed(range(self.horizon_length)):
            r_t = mb_rewards[t]
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones.float()
                # (num_envs, )
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                # (num_envs, )
            nextnonterminal = nextnonterminal.unsqueeze(1)
            # (num_envs, 1)

            # Calculate G_t = r_t + gamma * G_{t+1} * (1 - d_{t+1})
            mb_rewards_to_go[t] = r_t + self.gamma * g_t_plus_1 * nextnonterminal

            g_t_plus_1 = mb_rewards_to_go[t]  # update G_{t+1} for the next iteration

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        # if you use episodic returns, uncomment below
        # batch_dict['returns'] = mb_returns
        batch_dict['returns'] = swap_and_flatten01(mb_rewards_to_go).squeeze(-1)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict
        
    def play_steps(self):
        # collect task indices in `play_steps`
        batch_dict = self.inner_play_steps()
        # assuming `task_indices` unchanged during the time horizon length
        task_indices = self.all_task_indices
        arr = task_indices.unsqueeze(0).expand(self.horizon_length, -1)
        s = arr.size()
        batch_dict["task_indices"] = arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])
        return batch_dict
    
    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)
        task_indices = batch_dict["task_indices"]

        advantages = returns
        # (num_envs * horizon_length, )
        # normalize advantage by tasks
        for task_id in torch.unique(task_indices):
            mask = task_indices == task_id
            advantages[mask] = (advantages[mask] - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

        # if self.normalize_advantage:
        #     if self.is_rnn:
        #         if self.normalize_rms_advantage:
        #             advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
        #         else:
        #             advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
        #     else:
        #         if self.normalize_rms_advantage:
        #             advantages = self.advantage_mean_std(advantages)
        #         else:
        #             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # shuffle before adding to the dataset
        if self.shuffle_data:
            perm = torch.randperm(len(obses))
        else:
            perm = torch.arange(len(obses))
        dataset_dict = {}
        dataset_dict['old_logp_actions'] = neglogpacs[perm]
        
        # if you use episodic returns instead of reward-to-go, you need to repeat the advantages by the horizon length
        # we repeat the advantages by the horizon length b/c actor_loss_fn expects the advantage to be of shape (num_envs * horizon_length, )
        # dataset_dict['advantages'] = advantages.repeat_interleave(self.horizon_length)[perm]
        dataset_dict['advantages'] = advantages[perm]
        dataset_dict['actions'] = actions[perm]
        dataset_dict['obs'] = obses[perm]
        dataset_dict['dones'] = dones[perm]
        dataset_dict['rnn_states'] = rnn_states[perm] if rnn_states is not None else None
        dataset_dict['rnn_masks'] = rnn_masks[perm] if rnn_masks is not None else None
        dataset_dict['mu'] = mus[perm]
        dataset_dict['sigma'] = sigmas[perm]

        self.dataset.update_values_dict(dataset_dict)

        self.dataset.values_dict["task_indices"] = batch_dict["task_indices"][perm]

    def env_reset(self):
        obs = self.vec_env.reset()
        obs.update({
            "task_indices": self.vec_env.env.extras["task_indices"]
        })
        obs = self.obs_to_tensors(obs)
        return obs

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)
        obs.update({
            "task_indices": self.vec_env.env.extras["task_indices"]
        })

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos
        
    def backward(self, a_loss, entropy, b_loss, task_indices):
        # this is equally weighting grads from all the tasks
        loss = a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
        self.scaler.scale(loss.mean()).backward()

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states,
            'task_indices': obs['task_indices']
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
        return res_dict
    
    def compute_loss(self, input_dict):
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        actions_batch = input_dict['actions']
        task_indices = input_dict['task_indices']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
            'task_indices': task_indices
        }        

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            # import ipdb; ipdb.set_trace()
            action_log_probs = res_dict['prev_neglogp']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip, task_indices)

            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            loss = a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

        return loss, (mu, sigma, action_log_probs), (a_loss, entropy, b_loss)
    
    def calc_gradients(self, input_dict):
        old_action_log_probs_batch = input_dict['old_logp_actions']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        task_indices = input_dict['task_indices']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        
        lr_mul = 1.0
        curr_e_clip = self.e_clip

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']

        loss, (mu, sigma, action_log_probs), (a_loss, entropy, b_loss) = self.compute_loss(input_dict)
            
        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None
        
        self.backward(a_loss, entropy, b_loss, task_indices)
        # self.scaler.scale(loss).backward()
        
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.diagnostics.mini_batch(self,
        {
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        self.train_result = (a_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)

    def train_epoch(self):
        # even though it is PPO like
        # the algorithm does not update on the same batch of data more than once
        # We need to figure out a way to shuffle the data
        self.vec_env.set_train_info(self.frame, self)

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        b_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    av_kls = kl
                    if self.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.world_size
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.world_size
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                for k in range(len(self.model.running_mean_stds)):
                  self.model.running_mean_stds[k].eval()
                # self.model.running_mean_std.eval()

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, [torch.zeros_like(a_losses[0])]*len(a_losses), b_losses, entropies, kls, last_lr, lr_mul
    
class FAMOGRPOAgent(MTGRPOAgent):
    def __init__(
        self,
        base_name,
        params,
    ):
        super().__init__(base_name, params)
        task_ids = torch.unique(self.vec_env.env.extras["task_indices"])
        self.n_tasks = len(task_ids)
        self.num_global_updates = 0

        self.g = params["config"]["famo"]["gamma"]
        self.w_lr = params["config"]["famo"]["w_lr"]
        self.eps = params["config"]["famo"]["epsilon"]
        self.norm_w_grad = params["config"]["famo"]["norm_w_grad"]

        self.min_losses = torch.zeros(self.n_tasks).to(self.ppo_device)
        self.prev_loss = torch.zeros(self.n_tasks).to(self.ppo_device)
        self.w = torch.tensor([0.0] * self.n_tasks, device=self.ppo_device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=self.w_lr, weight_decay=self.g)

        self.epsilon = 1e-3

    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses):

        # print(["Losses: "] + [f"{n:.2g}" for n in losses.detach().cpu().numpy().tolist()])
        # print(["Prev Losses: "] + [f"{n:.2g}" for n in self.prev_loss.detach().cpu().numpy().tolist()])

        self.delta = (self.prev_loss - self.min_losses + self.eps).log() - \
                     (losses         - self.min_losses + self.eps).log()

        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + self.eps
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        # print("===================")
        # print("weights", z.detach().cpu().numpy())
        # print("logits", self.w.detach().cpu().numpy())

        if self.num_global_updates % 2000 == 0:
        # save the conflict matrix in debug folder
            from matplotlib import pyplot as plt
            if not os.path.exists("debug/famo"):
                os.makedirs("debug/famo")
            # barplot of the weights
            plt.bar(range(self.n_tasks), z.detach().cpu().numpy())
            plt.savefig(f"debug/famo/weights_{self.num_global_updates}.png")
            plt.close()
            # barplot of the logits
            plt.bar(range(self.n_tasks), self.w.detach().cpu().numpy())
            plt.savefig(f"debug/famo/logits_{self.num_global_updates}.png")
            plt.close()
            # barplot of the losses
            plt.bar(range(self.n_tasks), D.detach().cpu().numpy())
            plt.savefig(f"debug/famo/losses_{self.num_global_updates}.png")
            plt.close()
            # barplot of delta
            plt.bar(range(self.n_tasks), self.delta.detach().cpu().numpy())
            plt.savefig(f"debug/famo/delta_{self.num_global_updates}.png")
            plt.close()
        return loss
    
    def update(self):  #, curr_loss):
        # self.delta = (self.prev_loss - self.min_losses + self.eps).log() - \
        #              (curr_loss      - self.min_losses + self.eps).log()
        
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=self.delta.detach())[0]
        self.w_opt.zero_grad()
        if self.norm_w_grad:
            self.w.grad = d / torch.norm(d)
        else:
            self.w.grad = d
        self.w_opt.step()

        # print(["Delta: "] + [f"{n:.2g}" for n in self.delta.detach().cpu().numpy().tolist()])
        # print(["Weights: "] + [f"{n:.2g}" for n in F.softmax(self.w, -1).detach().cpu().numpy().tolist()])
        # print(["Gradient: "] + [f"{n:.2g}" for n in d.detach().cpu().numpy().tolist()], end="\n\n")

        self.num_global_updates += 1
        # self.prev_loss = curr_loss
    
    def backward(
        self,
        a_loss: torch.Tensor,
        entropy: torch.Tensor,
        b_loss: torch.Tensor,
        task_indices: torch.Tensor,
    ):
        total_loss = a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

        tids = torch.unique(task_indices)
        assert len(tids) == self.n_tasks, "Current batch does not contains all the tasks, it has only {}".format(tids)
        a_losses = []
        for tid in tids:
            mask = task_indices == tid
            a_losses.append((total_loss[mask].mean()))
        losses = torch.stack(a_losses)
        # make losses be non-negative
        # losses = losses - torch.min(losses) + self.epsilon # don't use in-place op!
        losses = torch.exp(losses)
        loss = self.get_weighted_loss(losses)
        
        self.scaler.scale(loss).backward()
        # self.scaler.scale(loss.mean()).backward()

        # actor_parameters = []
        # other_parameters = []
        # for n, p in self.model.named_parameters():
        #     if 'critic' in n or 'value' in n:
        #         other_parameters.append(p)
        #     else:
        #         actor_parameters.append(p)

        # GTG_a, w_cpu_a = famo_backward(a_loss, [entropy, b_loss], task_indices, actor_parameters)

    def calc_gradients(self, input_dict):
        super().calc_gradients(input_dict)

        self.update()  # torch.stack(c_losses))