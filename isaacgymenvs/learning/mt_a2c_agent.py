from typing import List, Tuple

import random
import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml

import torch 
from torch import nn
from torch import optim
import torch.distributed as dist
from torch.nn import functional as F
from scipy.optimize import minimize

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.common import a2c_common
from rl_games.common import common_losses
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common import datasets

from .grad_mani import pcgrad_backward, cagrad_backward
from .utils import filter_leader, print_statistics

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


def actor_loss_mt(old_action_neglog_probs_batch, action_neglog_probs, advantage, is_ppo, curr_e_clip, task_indices):
    # not sure if this advantage is normalized or not, but we can normlaize it again by tasks
    # normalize advantage by tasks
    for task_id in torch.unique(task_indices):
        mask = task_indices == task_id
        advantage[mask] = (advantage[mask] - advantage[mask].mean()) / (advantage[mask].std() + 1e-8)

    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
    else:
        a_loss = (action_neglog_probs * advantage)

    return a_loss


class MTA2CAgent(A2CAgent):
    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        self._device = self.config.get('device', 'cuda:0')
        self.shuffle_data = self.config.get('shuffle_data', False)

        self.all_task_indices : torch.Tensor = self.vec_env.env.extras["task_indices"]
        ordered_task_names : list[str] = self.vec_env.env.extras["ordered_task_names"]
        
        # this is the arguments for building the network
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

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_length' : self.seq_length,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        # change it to list of mean_std models
        if self.normalize_value:
            self.value_mean_stds = self.central_value_net.model.value_mean_stds if self.has_central_value else self.model.value_mean_stds

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)
        self.actor_loss_func = actor_loss_mt

    def restore(self, fn):
        # weights = torch.load(fn, map_location=self.device)
        weights = torch.load(fn)
        self.set_weights(weights)

    def set_weights(self, weights):
        new_weight_dict = {}
        for k in weights['model']:
            if "value_mean_stds" in k and "critic" in k:  # don't load critic and running mean of the value
                continue
            else:
                new_weight_dict[k] = weights['model'][k]
        self.model.load_state_dict(new_weight_dict, strict=False)
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(
                weights['running_mean_std'])

    def inner_play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs) # calls the model to sample an action (is_train=False)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

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

        last_values = self.get_values(self.obs)


        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict
        
    def play_steps(self):
        # collect task indices in `play_steps`
        batch_dict = self.inner_play_steps()
        # assuming `task_indices` unchanged during the time horizon length
        task_indices = self.vec_env.env.extras["task_indices"]
        arr = task_indices.unsqueeze(0).expand(self.horizon_length, -1)
        s = arr.size()
        batch_dict["task_indices"] = arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:]) 
        return batch_dict
    
    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)
        task_indices = batch_dict["task_indices"]

        advantages = returns - values

        if self.normalize_value:
            for tid in torch.unique(task_indices):
                self.value_mean_stds[tid.item()].train()
                mask = task_indices == tid
                values[mask] = self.value_mean_stds[tid.item()](values[mask])
                returns[mask] = self.value_mean_stds[tid.item()](returns[mask])
                self.value_mean_stds[tid.item()].eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    if os.getenv("LOCAL_RANK") and os.getenv("WORLD_SIZE"):
                        mean, var, _ = torch_ext.dist_mean_var_count(advantages.mean(), advantages.var(), len(advantages))
                        std = torch.sqrt(var)
                    else:
                        mean, std = advantages.mean(), advantages.std()
                    advantages = (advantages - mean) / (std + 1e-8)

        # shuffle before adding to the dataset
        if self.shuffle_data:
            perm = torch.randperm(len(advantages))
        else:
            perm = torch.arange(len(advantages))
        dataset_dict = {}
        dataset_dict['old_values'] = values[perm]
        dataset_dict['old_logp_actions'] = neglogpacs[perm]
        dataset_dict['advantages'] = advantages[perm]
        dataset_dict['returns'] = returns[perm]
        dataset_dict['actions'] = actions[perm]
        dataset_dict['obs'] = obses[perm]
        dataset_dict['dones'] = dones[perm]
        dataset_dict['rnn_states'] = rnn_states[perm] if rnn_states is not None else None
        dataset_dict['rnn_masks'] = rnn_masks[perm] if rnn_masks is not None else None
        dataset_dict['mu'] = mus[perm]
        dataset_dict['sigma'] = sigmas[perm]

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values[perm]
            dataset_dict['advantages'] = advantages[perm]
            dataset_dict['returns'] = returns[perm]
            dataset_dict['actions'] = actions[perm]
            dataset_dict['obs'] = batch_dict['states'][perm]
            dataset_dict['dones'] = dones[perm]
            dataset_dict['rnn_masks'] = rnn_masks[perm]
            self.central_value_net.update_dataset(dataset_dict)

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
        
    def get_values(self, obs):
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    'actions' : None,
                    'is_done': self.dones,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs['obs'])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : processed_obs,
                    'rnn_states' : self.rnn_states,
                    'task_indices': obs['task_indices']
                }
                result = self.model(input_dict)
                value = result['values']
            return value
        
    def backward(self, a_loss, c_loss, entropy, b_loss, task_indices):
        # this is equally weight grads from all the tasks
        loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
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
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict
    
    def compute_loss(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        return_batch = input_dict['returns']
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
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip, task_indices)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value).squeeze(-1)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

        return loss, (mu, sigma, action_log_probs), (a_loss, c_loss, entropy, b_loss)
    
    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        task_indices = input_dict['task_indices']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        
        lr_mul = 1.0
        curr_e_clip = self.e_clip

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']

        loss, (mu, sigma, action_log_probs), (a_loss, c_loss, entropy, b_loss) = self.compute_loss(input_dict)
            
        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None
        
        self.backward(a_loss, c_loss, entropy, b_loss, task_indices)
        # self.scaler.scale(loss).backward()
        
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()

        # import ipdb ; ipdb.set_trace()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        self.train_result = (a_loss, c_loss, entropy, \
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
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
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

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul


class PCGradA2CAgent(MTA2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.manipulate_actor_gradient = params["config"]["pcgrad"].get("manipulate_actor_gradient", False)
        self.manipulate_critic_gradient = params["config"]["pcgrad"].get("manipulate_critic_gradient", True)
        self.extra_loss_task = params["config"]["pcgrad"].get("extra_loss_task", False)

    def backward(self, a_loss, c_loss, entropy, b_loss, task_indices):
        critic_parameters = []
        other_parameters = []
        for n, p in self.model.named_parameters():
            if 'critic' in n or 'value' in n:
                critic_parameters.append(p)
            else:
                other_parameters.append(p)

        if self.manipulate_actor_gradient:
            if self.extra_loss_task:
                pcgrad_backward(a_loss, [entropy, b_loss], task_indices, other_parameters)
            else:
                pcgrad_backward(a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef,
                     [], task_indices, other_parameters)
        else:
            torch.mean(a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef).backward()
                
        if self.manipulate_critic_gradient:
            pcgrad_backward(c_loss, [], task_indices, critic_parameters)
        else:
            torch.mean(c_loss).backward()


class CAGradA2CAgent(MTA2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.manipulate_actor_gradient = params["config"]["cagrad"].get("manipulate_actor_gradient", True)
        self.manipulate_critic_gradient = params["config"]["cagrad"].get("manipulate_critic_gradient", True)
        self.extra_loss_task = params["config"]["cagrad"].get("extra_loss_task", False)
        self.c = params["config"]["cagrad"].get("c", 0.4)
        self.num_global_updates = 0
        self.timestamped_folder = f"debug/cagrad/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if not os.path.exists(self.timestamped_folder):
            os.makedirs(self.timestamped_folder)

    def backward(self, a_loss, c_loss, entropy, b_loss, task_indices):
        critic_parameters = []
        other_parameters = []
        for n, p in self.model.named_parameters():
            if 'critic' in n or 'value' in n:
                critic_parameters.append(p)
            else:
                other_parameters.append(p)

        if self.manipulate_actor_gradient:
            if self.extra_loss_task:
                GTG_a, w_cpu_a = cagrad_backward(a_loss, [entropy, b_loss], task_indices, other_parameters, c=self.c)
            else:
                GTG_a, w_cpu_a = cagrad_backward(a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef,
                     [], task_indices, other_parameters, c=self.c)
        else:
            torch.mean(a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef).backward()
            GTG_a, w_cpu_a = None, None

        if self.manipulate_critic_gradient:
            GTG_c, w_cpu_c = cagrad_backward(c_loss, [], task_indices, critic_parameters, c=self.c)
        else:
            torch.mean(c_loss).backward()
            GTG_c, w_cpu_c = None, None

        if self.num_global_updates % 2000 == 0:
            # save the conflict matrix in debug folder
            from matplotlib import pyplot as plt
            if GTG_a is not None:
                GTG_a = GTG_a
                GTG_a = GTG_a / np.matmul(np.diag(GTG_a)[:, None], np.diag(GTG_a)[None, :]) ** 0.5
                np.savetxt(os.path.join(self.timestamped_folder, f"GTG_a_{self.num_global_updates}.txt"), GTG_a)
                plt.imshow(GTG_a, vmin=-1, vmax=1)
                plt.colorbar()
                plt.savefig(os.path.join(self.timestamped_folder, f"GTG_a_{self.num_global_updates}.png"))
                plt.close()
            if GTG_c is not None:
                GTG_c = GTG_c
                GTG_c = GTG_c / np.matmul(np.diag(GTG_c)[:, None], np.diag(GTG_c)[None, :]) ** 0.5
                np.savetxt(os.path.join(self.timestamped_folder, f"GTG_c_{self.num_global_updates}.txt"), GTG_c)
                plt.imshow(GTG_c, vmin=-1, vmax=1)
                plt.colorbar()
                plt.savefig(os.path.join(self.timestamped_folder, f"GTG_c_{self.num_global_updates}.png"))
                plt.close()
        self.num_global_updates += 1


class FAMOA2CAgent(MTA2CAgent):
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

        self.debug_save_folder = f"debug/famo/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if not os.path.exists(self.debug_save_folder):
            os.makedirs(self.debug_save_folder)

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
            # barplot of the weights
            plt.bar(range(self.n_tasks), z.detach().cpu().numpy())
            plt.savefig(os.path.join(self.debug_save_folder, f"weights_{self.num_global_updates}.png"))
            plt.close()
            # barplot of the logits
            plt.bar(range(self.n_tasks), self.w.detach().cpu().numpy())
            plt.savefig(os.path.join(self.debug_save_folder, f"logits_{self.num_global_updates}.png"))
            plt.close()
            # barplot of the losses
            plt.bar(range(self.n_tasks), D.detach().cpu().numpy())
            plt.savefig(os.path.join(self.debug_save_folder, f"losses_{self.num_global_updates}.png"))
            plt.close()
            # barplot of delta
            plt.bar(range(self.n_tasks), self.delta.detach().cpu().numpy())
            plt.savefig(os.path.join(self.debug_save_folder, f"delta_{self.num_global_updates}.png"))
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
        c_loss: torch.Tensor,
        entropy: torch.Tensor,
        b_loss: torch.Tensor,
        task_indices: torch.Tensor,
    ):
        loss = a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
        self.scaler.scale(loss.mean()).backward()

        tids = torch.unique(task_indices)
        assert len(tids) == self.n_tasks, "Current batch does not contains all the tasks, it has only {}".format(tids)
        c_losses = []
        for tid in tids:
            mask = task_indices == tid
            c_losses.append((c_loss[mask].mean()) * 0.5 * self.critic_coef)
        c_loss = self.get_weighted_loss(losses=torch.stack(c_losses))
        self.scaler.scale(c_loss).backward()

    def calc_gradients(self, input_dict):
        super().calc_gradients(input_dict)

        # task_indices = input_dict['task_indices']
        # tids = torch.unique(task_indices)
        # with torch.no_grad():
        #     _, _, (a_loss, c_loss, entropy, b_loss) = self.compute_loss(input_dict)
        #     c_losses = []
        #     for tid in tids:
        #         mask = task_indices == tid
        #         c_losses.append(c_loss[mask].mean())

        self.update()  # torch.stack(c_losses))

# class PCGradA2CAgent(MTA2CAgent):
#     def __init__(self, base_name, params):
#         self.project_actor_gradient = params["config"]["pcgrad"].get("project_actor_gradient", False)
#         super().__init__(base_name, params)

#     def backward(self, a_loss, c_loss, entropy, b_loss, task_indices):
#         # only manipulate critic loss
#         critic_parameters = []
#         other_parameters = []
#         for n, p in self.model.named_parameters():
#             if 'critic' in n or 'value' in n:
#                 critic_parameters.append(p)
#             else:
#                 other_parameters.append(p)
        

#         if self.project_actor_gradient:
#             loss = a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef + c_loss * 0.5 * self.critic_coef
#         else:
#             loss = a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
#             self.scaler.scale(loss.mean()).backward()
#             loss = c_loss * 0.5 * self.critic_coef
#         tids = torch.unique(task_indices)
#         self.n_tasks = len(tids)
#         losses = []
#         for tid in tids:
#             mask = task_indices == tid
#             losses.append(self.scaler.scale(loss[mask].mean()))
#         self._set_pc_grads(losses, critic_parameters)
    
#         self.scaler.scale(loss.mean()).backward()

        

#     def _set_pc_grads(self, losses, shared_parameters):
#         # shared part
#         shared_grads = []
#         for l in losses:
#             shared_grads.append(
#                 torch.autograd.grad(l, shared_parameters, retain_graph=True)
#             )

#         non_conflict_shared_grads = self._project_conflicting(shared_grads)

#         for p, g in zip(shared_parameters, non_conflict_shared_grads):
#             p.grad = g
        
#     def _project_conflicting(self, grads: List[Tuple[torch.Tensor]]):
#         pc_grad = copy.deepcopy(grads)
#         for g_i in pc_grad:
#             random.shuffle(grads)
#             for g_j in grads:
#                 g_i_g_j = sum(
#                     [
#                         torch.dot(torch.flatten(grad_i), torch.flatten(grad_j))
#                         for grad_i, grad_j in zip(g_i, g_j)
#                     ]
#                 )
#                 if g_i_g_j < 0:
#                     g_j_norm_square = (
#                         torch.norm(torch.cat([torch.flatten(g) for g in g_j])) ** 2
#                     )
#                     for grad_i, grad_j in zip(g_i, g_j):
#                         grad_i -= g_i_g_j * grad_j / g_j_norm_square

#         merged_grad = [sum(g) for g in zip(*pc_grad)]
#         # by default use reduction mean
#         merged_grad = [g / self.n_tasks for g in merged_grad]

#         return merged_grad


# class CAGradA2CAgent(MTA2CAgent):
#     def __init__(self, base_name, params):
#         self.cagrad_cfg = params["config"].get("cagrad", {})
#         self.c = self.cagrad_cfg.get("c", 0.4)
#         self.operate_actor_gradient = self.cagrad_cfg.get("operate_actor_gradient", False)
#         super().__init__(base_name, params)

#     def backward(self, a_loss, c_loss, entropy, b_loss, task_indices):
#         # only manipulate critic loss
#         critic_parameters = []
#         other_parameters = []
#         for n, p in self.model.named_parameters():
#             if 'critic' in n or 'value' in n:
#                 critic_parameters.append(p)
#             else:
#                 other_parameters.append(p)

#         if self.operate_actor_gradient:
#             loss = a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef + c_loss * 0.5 * self.critic_coef
#             shared_parameters = critic_parameters + other_parameters
#         else:
#             loss = a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
#             self.scaler.scale(loss.mean()).backward()
#             loss = c_loss * 0.5 * self.critic_coef
#             shared_parameters = critic_parameters
#         tids = torch.unique(task_indices)
#         self.n_tasks = len(tids)
#         losses = []
#         for tid in tids:
#             mask = task_indices == tid
#             losses.append(self.scaler.scale(loss[mask].mean()))
            
#         self.get_weighted_loss(losses, shared_parameters)

#     def get_weighted_loss(
#         self,
#         losses,
#         shared_parameters,
#         **kwargs,
#     ):
#         """
#         Parameters
#         ----------
#         losses :
#         shared_parameters : shared parameters
#         kwargs :
#         Returns
#         -------
#         """
#         # NOTE: we allow only shared params for now. Need to see paper for other options.
#         grad_dims = []
#         for param in shared_parameters:
#             grad_dims.append(param.data.numel())
#         grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

#         for i in range(self.n_tasks):
#             if i < self.n_tasks:
#                 self.scaler.scale(losses[i].mean()).backward(retain_graph=True)
#             else:
#                 self.scaler.scale(losses[i].mean()).backward()
#             self.grad2vec(shared_parameters, grads, grad_dims, i)
#             # multi_task_model.zero_grad_shared_modules()
#             for p in shared_parameters:
#                 p.grad = None

#         g, GTG, w_cpu = self.cagrad(grads, alpha=self.c, rescale=1)
#         self.overwrite_grad(shared_parameters, g, grad_dims)
#         return GTG, w_cpu

#     def cagrad(self, grads, alpha=0.5, rescale=1):
#         GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
#         g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

#         x_start = np.ones(self.n_tasks) / self.n_tasks
#         bnds = tuple((0, 1) for x in x_start)
#         cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
#         A = GG.numpy()
#         b = x_start.copy()
#         c = (alpha * g0_norm + 1e-8).item()

#         def objfn(x):
#             return (
#                 x.reshape(1, self.n_tasks).dot(A).dot(b.reshape(self.n_tasks, 1))
#                 + c
#                 * np.sqrt(
#                     x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1))
#                     + 1e-8
#                 )
#             ).sum()

#         res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
#         w_cpu = res.x
#         ww = torch.Tensor(w_cpu).to(grads.device)
#         gw = (grads * ww.view(1, -1)).sum(1)
#         gw_norm = gw.norm()
#         lmbda = c / (gw_norm + 1e-8)
#         g = grads.mean(1) + lmbda * gw
#         if rescale == 0:
#             return g, GG.numpy(), w_cpu
#         elif rescale == 1:
#             return g / (1 + alpha ** 2), GG.numpy(), w_cpu
#         else:
#             return g / (1 + alpha), GG.numpy(), w_cpu

#     @staticmethod
#     def grad2vec(shared_params, grads, grad_dims, task):
#         # store the gradients
#         grads[:, task].fill_(0.0)
#         cnt = 0
#         # for mm in m.shared_modules():
#         #     for p in mm.parameters():

#         for param in shared_params:
#             grad = param.grad
#             if grad is not None:
#                 grad_cur = grad.data.detach().clone()
#                 beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
#                 en = sum(grad_dims[: cnt + 1])
#                 grads[beg:en, task].copy_(grad_cur.data.view(-1))
#             cnt += 1

#     def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
#         newgrad = newgrad * self.n_tasks  # to match the sum loss
#         cnt = 0

#         # for mm in m.shared_modules():
#         #     for param in mm.parameters():
#         for param in shared_parameters:
#             beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
#             en = sum(grad_dims[: cnt + 1])
#             this_grad = newgrad[beg:en].contiguous().view(param.data.size())
#             param.grad = this_grad.data.clone()
#             cnt += 1