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

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.common import a2c_common
from rl_games.common import common_losses
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

class MLA2CAgent(A2CAgent):
    """ Meta-learning agent
    """
    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        self.optimizer_state = None
        self.outer_lr = 5e-3

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

        self.train_task_datasets = {}
        self.test_task_datasets = {}
        self.train_tasks = None

        # list of mean_std models
        if self.normalize_value:
            self.value_mean_stds = self.central_value_net.model.value_mean_stds if self.has_central_value else self.model.value_mean_stds

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)
        self.shuffle_data = False

        self.meta_batch_size = self.vec_env.env.meta_batch_size
        self.num_envs_per_task = self.vec_env.env.num_envs_per_task

        batch_size = self.num_agents * self.num_actors
        current_rewards_shape = (batch_size, self.value_size)
        self.test_current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.test_current_shaped_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.test_current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.test_dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

        self.test_game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.test_game_shaped_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.test_game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)

    def update_module_params(self, module, new_params):
        """Load parameters to a module.

        This function acts like `torch.nn.Module._load_from_state_dict()`, but
        it replaces the tensors in module with those in new_params, while
        `_load_from_state_dict()` loads only the value. Use this function so
        that the `grad` and `grad_fn` of `new_params` can be restored

        Args:
            module (torch.nn.Module): A torch module.
            new_params (dict): A dict of torch tensor used as the new
                parameters of this module. This parameters dict should be
                generated by `torch.nn.Module.named_parameters()`

        """
        named_modules = dict(module.named_modules())

        def update(m, name, param):
            del m._parameters[name]
            setattr(m, name, param)
            m._parameters[name] = param

        for name, new_param in new_params.items():
            if '.' in name:
                module_name, param_name = tuple(name.rsplit('.', 1))
                if module_name in named_modules:
                    update(named_modules[module_name], param_name, new_param)
            else:
                update(module, name, new_param)
        
    def play_steps(self):
        self.obs = self.env_reset()
        # collect task indices in `play_steps`
        batch_dict = super().play_steps()
        # assuming `task_indices` unchanged during the time horizon length
        task_indices = torch.flatten(torch.tensor([[task_idx]*self.num_envs_per_task for task_idx in self.train_tasks],device=self.ppo_device))
        arr = task_indices.unsqueeze(0).expand(self.horizon_length, -1)
        s = arr.size()
        batch_dict["task_indices"] = arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])
        return batch_dict
    
    def extract_task_data(self,dataset_dict,i):
        num_envs = self.vec_env.env.num_envs
        attributes = ['old_values','old_logp_actions','advantages','returns','actions','obs','dones','mu','sigma','task_indices']
        task_dic = {k:None for k in attributes}

        for k in attributes:
            # unflatten and transpose to undo transformation in play_steps
            # reshaped_data = dataset_dict[k].view(num_envs,self.horizon_length,-1).transpose(0,1)
            task_dic[k] = dataset_dict[k].view(num_envs,self.horizon_length,-1).transpose(0,1)[:,i*self.num_envs_per_task:(i+1)*self.num_envs_per_task,:]
            # for h in range(self.horizon_length):
            #     task_dic[k].append(reshaped_data[h,i*self.num_envs_per_task:(i+1)*self.num_envs_per_task])

        # get random ordering
        # if self.shuffle_data:
        #     perm = torch.randperm(self.num_envs_per_task*self.horizon_length)
        # else:
        #     perm = torch.arange(self.num_envs_per_task*self.horizon_length)
        
        for k in attributes:
            # merge across the horizon length
            # task_dic[k] = torch.cat(task_dic[k],dim=0)
            # if k == 'advantages' and self.normalize_advantage:
            #     # normalize advantage by task
            #     task_dic['advantages'] = (task_dic['advantages'] - task_dic['advantages'].mean()) / (task_dic['advantages'].std() + 1e-8)
            # swap and flatten to match play_steps()
            # task_dic[k] = task_dic[k].reshape(self.horizon_length,self.num_envs_per_task,-1).transpose(0,1).reshape(self.horizon_length*self.num_envs_per_task,-1)
            s = task_dic[k].size()
            task_dic[k] = task_dic[k].transpose(0,1).reshape(s[0]*s[1],*s[2:])

        # remove extra dim on some attributes
        task_dic['old_logp_actions'] = task_dic['old_logp_actions'].squeeze(-1)
        task_dic['advantages'] = task_dic['advantages'].squeeze(-1)
        task_dic['task_indices'] = task_dic['task_indices'].squeeze(-1)

        return task_dic
    
    def prepare_dataset(self, batch_dict, training_mode=True):
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
            # self.value_mean_std.train()
            # values = self.value_mean_std(values)
            # returns = self.value_mean_std(returns)
            # self.value_mean_std.eval()
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
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
        dataset_dict['task_indices'] = task_indices[perm]

        # self.dataset.update_values_dict(dataset_dict)

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

        # self.dataset.values_dict["task_indices"] = batch_dict["task_indices"][perm]

        batch_size = self.horizon_length * self.num_envs_per_task 
        mini_batch_size = 4000

        if training_mode:
            for i, task_idx in enumerate(self.train_tasks):
                self.train_task_datasets[task_idx] = datasets.PPODataset(batch_size, mini_batch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
                self.train_task_datasets[task_idx].update_values_dict(self.extract_task_data(dataset_dict, i))
        else:
            for i, task_idx in enumerate(self.train_tasks):
                self.test_task_datasets[task_idx] = datasets.PPODataset(batch_size, mini_batch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
                self.test_task_datasets[task_idx].update_values_dict(self.extract_task_data(dataset_dict, i))

    def env_reset(self,eval_task=None):
        obs = self.vec_env.reset()
        if self.train_tasks is not None:
            if eval_task is None:
                obs.update({
                    "task_indices": torch.flatten(torch.tensor([[task_idx]*self.num_envs_per_task for task_idx in self.train_tasks],device=self.ppo_device))
                })
            else:
                obs.update({
                    "task_indices": torch.flatten(torch.tensor([eval_task]*self.num_actors,device=self.ppo_device))
                })
        obs = self.obs_to_tensors(obs)
        return obs

    def env_step(self, actions,eval_task=None):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)
        if eval_task is None:
            obs.update({
                "task_indices": torch.flatten(torch.tensor([[task_idx]*self.num_envs_per_task for task_idx in self.train_tasks],device=self.ppo_device))
            })
        else:
            obs.update({
                "task_indices": torch.flatten(torch.tensor([eval_task]*self.num_actors,device=self.ppo_device))
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
        # this equally weights grads
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

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

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
    
    def trancate_gradients_and_step(self):
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))

            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.world_size
                    )
                    offset += param.numel()

        if self.truncate_grads:
            self.scaler.unscale_(self.inner_optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.scaler.step(self.inner_optimizer)
        self.scaler.update()
    
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
            self.inner_optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None
        
        self.backward(a_loss, c_loss, entropy, b_loss, None)
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
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)
        
    def sample_tasks(self,num_total_tasks=50):
        """Sample which 50 training tasks to use for the current iteration and resets the environment accordingly
        """
        self.train_tasks = self.vec_env.env.set_meta_parameters(num_total_tasks)

    def obtain_samples(self):
        """Obtain samples for each task of the form (horizon_limit*num_envs, -1) using play_steps before adaptation
        """
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

        return play_time_start, play_time_end, update_time_start, batch_dict['step_time']
    
    def obtain_meta_samples(self,all_params):
        """Samples data using the adapted parameters of the model on each task
        """
        self.set_eval()
        play_time_start = time.time()
        for k in range(len(self.model.running_mean_stds)):
            self.model.running_mean_stds[k].train()
        with torch.no_grad():
            batch_dict = self.play_adapted_steps(all_params)
        play_time_end = time.time()
        
        self.set_train()
        self.curr_frames += batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict, training_mode=False)

        return play_time_end - play_time_start, batch_dict['step_time']
    
    
    def get_res_dict_from_obs(self, obs, all_params):
        """Computes action values from observation using the corresponding task parameters

        Returns a dictionary with the following keys: 'neglogpacs', 'values', 'actions', 'mus', 'sigmas'
        """
        actions = []
        res_dict = {k:[] for k in ['neglogpacs', 'values', 'actions', 'mus', 'sigmas']}

        for i,task_idx in enumerate(self.train_tasks):
            # switch to the adapted parameters for the current task
            self.update_module_params(self.model, all_params[task_idx])
            # get the last observations of the environments corresponding to the current task
            last_obs = {'obs':obs['obs'][i*self.num_envs_per_task:(i+1)*self.num_envs_per_task], \
                   'task_indices':obs['task_indices'][i*self.num_envs_per_task:(i+1)*self.num_envs_per_task].to(dtype=torch.int64)}
            # get the action values
            cur_dict = self.get_action_values(last_obs)

            for k in cur_dict:
                if cur_dict[k] is not None:
                    res_dict[k].append(cur_dict[k])
        
        for k in res_dict:
            res_dict[k] = torch.cat(res_dict[k],dim=0)
        return res_dict
    

    def play_adapted_steps(self, all_params):
        """Returns a dictionary that holds rollouts for each task's policy

        This function plays each task with its corresponding parameters
        for a fixed number of steps and returns the following attributes:
        'actions', 'neglogpacs', 'values', 'mus', 'sigmas', 'obses', 'dones', 'returns', 
        'played_frames', 'step_time', 'task_indices'
        
        Some important notes: - the ith row of actions correspond the action taken after observing the ith row of obses
                              - after env_step, self.obs stores the most recent observation of the environments, which has no action taken upon it. 
        """
        update_list = self.update_list

        step_time = 0.0
        self.obs = self.env_reset()

        for n in range(self.horizon_length):
            res_dict = self.get_res_dict_from_obs(self.obs, all_params)
            self.experience_buffer.update_data('obses', n, self.obs['obs']) # tensor_dict stores 'obs' as shape (horizon_length, num_envs, obs_size)
            self.experience_buffer.update_data('dones', n, self.dones)

            # put keys from res_dict - 'neglogpacs', 'values', 'actions', 'mus', 'sigmas' - into the experience buffer
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

        task_indices = torch.flatten(torch.tensor([[task_idx]*self.num_envs_per_task for task_idx in self.train_tasks],device=self.ppo_device))
        arr = task_indices.unsqueeze(0).expand(self.horizon_length, -1)
        s = arr.size()
        batch_dict["task_indices"] = arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

        return batch_dict

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.inner_optimizer.param_groups:
            param_group['lr'] = lr
        
        #if self.has_central_value:
        #    self.central_value_net.update_lr(lr)

    def update_outer_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        #if self.has_central_value:
        #    self.central_value_net.update_lr(lr)
    
    def adapt(self,initial_theta):
        """ Perform adaptation (fine-tuning) for each task and returns the adapted parameters of the model on each task
        """
        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []
        
        all_params = [None] * 50

        # save running obs stats
        # if self.normalize_value:
        #     initial_value_mean_std_states = [None] * 50
        #     for k in range(len(self.value_mean_stds)):
        #         initial_value_mean_std_states[k] = copy.deepcopy(self.value_mean_stds[k].state_dict())
        
        # # save running value stats
        # if self.normalize_input:
        #     initial_running_mean_std_states = [None] * 50
        #     for k in range(len(self.model.running_mean_stds)):
        #         initial_running_mean_std_states[k] = copy.deepcopy(self.model.running_mean_stds[k].state_dict())

        for task_idx in self.train_tasks:
            last_lr = 3e-4
            self.inner_optimizer = optim.SGD(self.model.parameters(), lr=last_lr)
            for mini_ep in range(0, self.mini_epochs_num):
                ep_kls = []
                for i in range(len(self.train_task_datasets[task_idx])):
                    a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.train_task_datasets[task_idx][i])
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    ep_kls.append(kl)
                    entropies.append(entropy)
                    if self.bounds_loss_coef is not None:
                        b_losses.append(b_loss)

                    self.train_task_datasets[task_idx].update_mu_sigma(cmu, csigma)

                av_kls = torch_ext.mean_list(ep_kls)
                if self.multi_gpu:
                    dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                    av_kls /= self.world_size
                # if self.schedule_type == 'standard':
                #     last_lr, self.entropy_coef = self.scheduler.update(last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                #     self.update_lr(last_lr)

                kls.append(av_kls)
                self.diagnostics.mini_epoch(self, mini_ep)
                if self.normalize_input:
                    self.model.running_mean_stds[task_idx].eval() # don't need to update statstics more than one mini epoch

            # save parameters of the model finetuned on each sampled task
            all_params[task_idx] = copy.deepcopy(dict(self.model.named_parameters()))
            # additionally, save gradients since deepcopy doesn't copy gradients
            for param_name, param_value in self.model.named_parameters():
                all_params[task_idx][param_name].grad = param_value.grad.clone()

            # reset the actor-critic network back to initial theta
            self.model.a2c_network.load_state_dict(initial_theta)
            for param in self.model.parameters():
                param.grad = None
            
        return all_params, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul
    
    def compute_meta_gradient(self, all_params):
        """ calculate meta-loss = 1/meta_batch size * \sum_i \nabla_\theta L_{\tau_i]}(\phi_i;D_test) using the chain rule ------------#
        #     
        #   = 1/meta_batch size * \sum_i \nabla_\phi L_{\tau_i}(\phi_i;D_test) @ \nabla_\theta \phi_i
        #   we approximate the second term as the identity matrix
        """          
        # stop the gradients at the adapted parameters (zeros out gradients of all parameters)
        for task_idx in self.train_tasks:
            for name,param in all_params[task_idx].items():
                all_params[task_idx][name] = all_params[task_idx][name].detach().requires_grad_(True)
        
        # calculate the first term - the loss of the test data at the adapted parameters (from PPO)    
        all_grads = [] # of length (meta_batch_size)
        for task_idx in self.train_tasks:
            self.update_module_params(self.model, all_params[task_idx])
            task_loss, *_ = self.compute_loss(self.test_task_datasets[task_idx][0])
            for i in range(1, len(self.test_task_datasets[task_idx])):
                loss, *_ = self.compute_loss(self.test_task_datasets[task_idx][i])
                task_loss += loss
            task_loss /= len(self.test_task_datasets[task_idx])
            all_grads.append(list(torch.autograd.grad(torch.mean(task_loss), self.model.parameters())))
            if self.normalize_input:
                self.model.running_mean_stds[task_idx].eval() # don't need to update statstics more than one mini epoch
        
        # get the average gradient across all tasks
        avg_grads = all_grads[0]
        for grad_idx in range(len(all_grads[0])):
            for i in range(1,self.meta_batch_size):
                avg_grads[grad_idx] += all_grads[i][grad_idx]
            avg_grads[grad_idx] /= self.meta_batch_size
        
        return avg_grads

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

        self.sample_tasks()
        self.env_reset()

        train_play_time_start, train_play_time_end, update_time_start, train_step_time = self.obtain_samples()

        initial_theta = copy.deepcopy(self.model.a2c_network.state_dict())
        
        all_params, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.adapt(initial_theta)

        meta_play_time, meta_step_time = self.obtain_meta_samples(all_params)

        avg_grads = self.compute_meta_gradient(all_params)

        # assign meta gradient to parameter grad and take optimization step
        self.model.a2c_network.load_state_dict(initial_theta)
        for (name,param),g in zip(dict(self.model.named_parameters()).items(), avg_grads):
            param.grad = g

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.outer_lr, eps=1e-08)

        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)

        # if self.schedule_type == 'standard':
        #     self.outer_lr, self.entropy_coef = self.scheduler.update(self.outer_lr, self.entropy_coef, self.epoch_num, 0, torch_ext.mean_list(kls))
        #     self.update_outer_lr(self.outer_lr)
        
        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.optimizer_state = copy.deepcopy(self.optimizer.state_dict())

        # cleaning memory to optimize space
        for task_idx in self.train_tasks:
            if self.train_task_datasets[task_idx]:
                self.train_task_datasets[task_idx].update_values_dict(None)
                self.test_task_datasets[task_idx].update_values_dict(None)

        update_time_end = time.time()
        play_time = (train_play_time_end - train_play_time_start) + meta_play_time
        update_time = update_time_end - update_time_start
        total_time = update_time_end - train_play_time_start
        
        step_time = meta_step_time + train_step_time
        return step_time, play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, self.outer_lr, lr_mul

class MLA2CReptileAgent(MLA2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        self.outer_lr = .05
        self.cosine_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.scheduler_state = None

    def sample_tasks(self,num_total_tasks=50,train=True):
        """Sample which 50 training tasks to use for the current iteration and resets the environment accordingly
        """
        train_tasks = self.vec_env.env.set_meta_parameters(train, num_total_tasks)
        return train_tasks

    def prepare_dataset(self, batch_dict, eval_task=None):
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
            # self.value_mean_std.train()
            # values = self.value_mean_std(values)
            # returns = self.value_mean_std(returns)
            # self.value_mean_std.eval()
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
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
        dataset_dict['task_indices'] = task_indices[perm]

        # self.dataset.update_values_dict(dataset_dict)

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

        # self.dataset.values_dict["task_indices"] = batch_dict["task_indices"][perm]

        batch_size = self.horizon_length * self.num_envs_per_task 
        mini_batch_size = self.minibatch_size

        if eval_task is None:
            for i, task_idx in enumerate(self.train_tasks):
                self.train_task_datasets[task_idx] = datasets.PPODataset(batch_size, mini_batch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
                self.train_task_datasets[task_idx].update_values_dict(self.extract_task_data(dataset_dict, i))
        else:
            self.test_task_datasets[eval_task] = datasets.PPODataset(batch_size, mini_batch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
            self.test_task_datasets[eval_task].update_values_dict(dataset_dict)
            self.test_task_datasets[eval_task].values_dict["task_indices"] = batch_dict["task_indices"][perm]

    def adapt(self,initial_theta):
        """ Perform adaptation (fine-tuning) for each task and returns the adapted parameters of the model on each task
        """
        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []
        
        phis = [None]*50

        for task_idx in self.train_tasks:  
            last_lr = self.last_lr
            self.inner_optimizer = optim.Adam(self.model.parameters(), lr=last_lr, eps=1e-08, weight_decay=self.weight_decay)
            for mini_ep in range(0, self.mini_epochs_num):
                ep_kls = []
                for i in range(len(self.train_task_datasets[task_idx])):
                    a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.train_task_datasets[task_idx][i])
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    ep_kls.append(kl)
                    entropies.append(entropy)
                    if self.bounds_loss_coef is not None:
                        b_losses.append(b_loss)

                    self.train_task_datasets[task_idx].update_mu_sigma(cmu, csigma)

                av_kls = torch_ext.mean_list(ep_kls)
                if self.multi_gpu:
                    dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                    av_kls /= self.world_size
                if self.schedule_type == 'standard':
                    last_lr, self.entropy_coef = self.scheduler.update(last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(last_lr)

                kls.append(av_kls)
                self.diagnostics.mini_epoch(self, mini_ep)
                if self.normalize_input:
                    self.model.running_mean_stds[task_idx].eval() # don't need to update statstics more than one mini epoch

            # save parameters of the model finetuned on each sampled task, no need to save gradients since we are using REPTILE
            phis[task_idx] = copy.deepcopy(dict(self.model.named_parameters()))

            # reset the actor-critic network back to initial theta
            self.model.a2c_network.load_state_dict(initial_theta)
            for param in self.model.parameters():
                param.grad = None
            
        return phis, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul

    def exponential_lr(self, outer_lr):
        if self.outer_lr > 5e-3:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.99
            return outer_lr*0.99
        return outer_lr
    
    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

        self.train_tasks = self.sample_tasks(num_total_tasks=50)
        self.env_reset()

        train_play_time_start, train_play_time_end, update_time_start, train_step_time = self.obtain_samples()

        initial_theta = copy.deepcopy(self.model.a2c_network.state_dict())
        
        phis, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.adapt(initial_theta)

        # assign meta gradient (phi - theta) as grads and take optimization step
        self.model.a2c_network.load_state_dict(initial_theta)
        for task_idx in self.train_tasks:
            for (name,param) in dict(self.model.named_parameters()).items():
                if param.grad is not None:
                    param.grad += - (phis[task_idx][name]-param) # invert to do descent
                else:
                    param.grad = - (phis[task_idx][name]-param)

        for (name,param) in dict(self.model.named_parameters()).items():
            param.grad /= len(self.train_tasks)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.outer_lr)
        # self.cosine_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        # if self.scheduler_state is not None:
        #     self.cosine_scheduler.load_state_dict(self.scheduler_state)
        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)
        
        self.outer_lr = self.exponential_lr(self.outer_lr)
        
        self.optimizer.step()
        # self.cosine_scheduler.step()

        last_theta = copy.deepcopy(self.model.a2c_network.state_dict())
        self.optimizer_state = copy.deepcopy(self.optimizer.state_dict())

        # cleaning memory to optimize space
        for task_idx in self.train_tasks:
            if self.train_task_datasets[task_idx]:
                self.train_task_datasets[task_idx].update_values_dict(None)
        
        if self.epoch_num % 50 == 0:
            self.eval_epoch()
        
        # evaluation will adapt model to eval_task, so it must be reset back to the theta from after the meta-optimization step
        self.model.a2c_network.load_state_dict(last_theta)

        update_time_end = time.time()
        play_time = (train_play_time_end - train_play_time_start)
        update_time = update_time_end - update_time_start
        total_time = update_time_end - train_play_time_start
        
        step_time = train_step_time
        return step_time, play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, self.outer_lr, lr_mul

    def play_eval_steps(self,eval_task,write_stats=False):
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            res_dict = self.get_action_values(self.obs) # calls the model to sample an action (is_train=False)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'],eval_task=eval_task)
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            if write_stats:
                self.test_current_rewards += rewards
                self.test_current_shaped_rewards += shaped_rewards
                self.test_current_lengths += 1
                all_done_indices = self.dones.nonzero(as_tuple=False)
                env_done_indices = all_done_indices[::self.num_agents]

                self.test_game_rewards.update(self.test_current_rewards[env_done_indices])
                self.test_game_shaped_rewards.update(self.test_current_shaped_rewards[env_done_indices])
                self.test_game_lengths.update(self.test_current_lengths[env_done_indices])
                # self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.test_current_rewards = self.test_current_rewards * not_dones.unsqueeze(1)
            self.test_current_shaped_rewards = self.test_current_shaped_rewards * not_dones.unsqueeze(1)
            self.test_current_lengths = self.test_current_lengths * not_dones

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

        # assuming `task_indices` unchanged during the time horizon length
        task_indices = torch.flatten(torch.tensor([eval_task]*self.num_actors,device=self.ppo_device))
        arr = task_indices.unsqueeze(0).expand(self.horizon_length, -1)
        s = arr.size()
        batch_dict["task_indices"] = arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

        if write_stats:
            mean_rewards = self.test_game_rewards.get_mean()
            mean_shaped_rewards = self.test_game_shaped_rewards.get_mean()
            mean_lengths = self.test_game_lengths.get_mean()
            print(f"{'*'*50} EVAL STATS on task {eval_task} {'*'*50}")
            print("REWARDS", mean_rewards[0])
            print("LENGTHS", mean_lengths)

        return batch_dict
    
    def eval_epoch(self):
        eval_task = self.sample_tasks(num_total_tasks=50,train=False)[0]
        self.obs = self.env_reset(eval_task=eval_task)
        batch_dict = self.play_eval_steps(eval_task)

        self.prepare_dataset(batch_dict, eval_task=eval_task)

        self.inner_optimizer = optim.Adam(self.model.parameters(), lr=self.last_lr, eps=1e-08, weight_decay=self.weight_decay)

        eval_epochs_num = 32

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        ep_kls = []
        kls = []
        
        last_lr = self.last_lr
        for mini_ep in range(0, eval_epochs_num):
            ep_kls = []
            for i in range(len(self.test_task_datasets[eval_task])):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.test_task_datasets[eval_task][i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.test_task_datasets[eval_task].update_mu_sigma(cmu, csigma)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.schedule_type == 'standard':
                last_lr, self.entropy_coef = self.scheduler.update(last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_stds[eval_task].eval()
        

        batch_dict = self.play_eval_steps(eval_task, write_stats=True)

        # cleaning memory to optimize space
        self.test_task_datasets[eval_task].update_values_dict(None)
