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

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.common import a2c_common
from rl_games.common import common_losses
from rl_games.common import datasets

from isaacgymenvs.learning.mt_a2c_agent import MTA2CAgent
from isaacgymenvs.learning.mt_models import PerTaskRewardNormalizer

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

def actor_loss_sapg_mt(old_action_neglog_probs_batch, action_neglog_probs, advantage, is_ppo, curr_e_clip, task_indices):
    if is_ppo:
        ratio = torch.exp(old_action_neglog_probs_batch - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
    else:
        a_loss = (action_neglog_probs * advantage)

    return a_loss

class SAPGA2CAgent(MTA2CAgent):
    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        self._device = self.config.get('device', 'cuda:0')
        self.shuffle_data = self.config.get('shuffle_data', True)

        self.normalize_reward = self.config.get('normalize_reward', False)

        self.all_task_indices : torch.Tensor = self.vec_env.env.extras["task_indices"].to(self.ppo_device)
        self.task_embedding_dim = torch.unique(self.all_task_indices).shape[0]
        ordered_task_names : list[str] = self.vec_env.env.extras["ordered_task_names"]

        self.use_leader_follower_aggregation = self.config.get('use_leader_follower_aggregation', True)
        self.num_tasks_per_leader = self.config.get('num_tasks_per_leader', 1)
        self.num_tasks_per_follower = self.config.get('num_tasks_per_follower', 1)

        self.block_size = self.config.get('block_size', 4096)
        self.num_policies = self.num_actors // self.block_size
        
        # Assign policies in a global round-robin fashion.
        policy_ids = torch.arange(self.num_actors, device=self.ppo_device) % self.num_policies
                
        self.policy_ids_flat = policy_ids
        self.policy_ids = policy_ids.reshape(-1, 1).float()

        self.policy_ids_one_hot = torch.nn.functional.one_hot(
            self.policy_ids.long().squeeze(-1), num_classes=self.num_policies
        ).float()

        input_shape = (obs_shape[0] + self.num_policies,)
        
        self.expl_reward_type = self.config.get('expl_reward_type', 'none')
        
        policy_coeffs = torch.zeros(self.num_policies, device=self.ppo_device)
        if self.expl_reward_type == 'per-policy':
            if self.num_tasks_per_follower != self.num_tasks_per_leader:
                follower_entropy_coef = self.config.get('expl_reward_coef_scale', 0.01)
                policy_coeffs.fill_(follower_entropy_coef)
                policy_coeffs[0] = 0.0
            else:
                start_val = 0.001
                end_val = self.config.get('expl_reward_coef_scale', 0.01) 
                policy_coeffs = torch.logspace(
                    np.log10(start_val), 
                    np.log10(end_val), 
                    self.num_policies, 
                    device=self.ppo_device
                )
                policy_coeffs[0] = 0.0
        elif self.expl_reward_type == 'none':
            policy_coeffs.fill_(self.entropy_coef)

        self.policy_coeffs = policy_coeffs

        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : input_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
            'task_indices': self.all_task_indices,
            'task_embedding_dim': torch.unique(self.all_task_indices).shape[0],
            'ordered_task_names': ordered_task_names,
            'device': self._device,
            'policy_embedding_dim' : self.config.get('policy_embedding_dim', 16),
            'num_policies': self.num_policies,
            'num_tasks_per_leader': self.config.get('num_tasks_per_leader', 1),
            'num_tasks_per_follower': self.config.get('num_tasks_per_follower', 1),
            'policy_ids':  torch.arange(self.num_policies, device=self.ppo_device),
        }
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound')

        self.optimizer = optim.Adam(
            self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay
        )

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 'num_agents' : self.num_agents,
                'horizon_length' : self.horizon_length, 'num_actors' : self.num_actors,
                'num_actions' : self.actions_num, 'seq_length' : self.seq_length,
                'normalize_value' : self.normalize_value, 'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 'writter' : self.writer,
                'max_epochs' : self.max_epochs, 'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.minibatch_size = self.config['minibatch_size']
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        
        if self.normalize_value:
            self.value_mean_stds = self.central_value_net.model.value_mean_stds if self.has_central_value else self.model.value_mean_stds

        if self.normalize_reward:
            self.reward_normalizer = PerTaskRewardNormalizer(torch.unique(self.all_task_indices).shape[0], self.gamma, device=self.ppo_device, g_max=5000).to(self.ppo_device)

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)
        self.actor_loss_func = actor_loss_sapg_mt
        self.off_policy_ratio = self.config.get('off_policy_ratio', 1.0)

        # Per-epoch exact success rate accumulators (reset at end of train_epoch).
        # Reported as Σ successes / Σ episodes within the epoch — hparam-invariant
        # definition, unlike Episode/average_environment_success_rate which is
        # per-step-weighted.
        self.policy_epoch_success_sum = torch.zeros(self.num_policies, device=self.ppo_device)
        self.policy_epoch_episode_count = torch.zeros(self.num_policies, device=self.ppo_device)

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        
        from rl_games.common.experience import ExperienceBuffer
        
        self.env_info['observation_space'] = spaces.Box(
            low=np.concatenate([self.env_info['observation_space'].low, np.zeros(self.num_policies)]),
            high=np.concatenate([self.env_info['observation_space'].high, np.ones(self.num_policies)]),
            dtype=np.float32
        )
        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)

        val_shape = (self.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)

        self.experience_buffer.tensor_dict['off_policy_value_leader'] = torch.zeros(
            val_shape, dtype=torch.float32, device=self.ppo_device
        )
        self.experience_buffer.tensor_dict['one_step_returns_leader'] = torch.zeros(
            val_shape, dtype=torch.float32, device=self.ppo_device
        )
        
        if not hasattr(self, 'current_rewards') or self.current_rewards is None:
            self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
            self.current_shaped_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
            self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
            self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)
        else:
            self.current_rewards = self.current_rewards.to(self.ppo_device)
            self.current_shaped_rewards = self.current_shaped_rewards.to(self.ppo_device)
            self.current_lengths = self.current_lengths.to(self.ppo_device)
            self.dones = self.dones.to(self.ppo_device)

        if self.is_rnn:
            if not hasattr(self, 'rnn_states') or self.rnn_states is None:
                self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

            total_agents = self.num_agents * self.num_actors
            num_seqs = self.horizon_length // self.seq_length
            assert((self.horizon_length * total_agents // self.num_minibatches) % self.seq_length == 0)
            self.mb_rnn_states = [torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.rnn_states]

        # Define the list of keys to update from acting policy's model output
        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        # Define the full list of keys needed for the final flattened batch dict
        self.tensor_list = [
            'obses', 'rewards', 'dones', 'actions', # Base rollout data
            'neglogpacs', 'values', 'mus', 'sigmas',               # Acting policy outputs
            'returns',                               # On-policy GAE results
            'off_policy_value_leader', 'one_step_returns_leader' # OFF-POLICY 1-step return targets
        ]

    def env_reset(self):
        obs = self.vec_env.reset()
        obs.update({
            "task_indices": self.vec_env.env.extras["task_indices"]
        })
        if self.policy_ids is not None:
            obs['obs'] = torch.cat([obs['obs'], self.policy_ids_one_hot], dim=-1)
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
            rewards, dones, infos = rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            rewards, dones, infos = torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos
        
        tr_obs = self.obs_to_tensors(obs)
        if self.policy_ids is not None:
            tr_obs['obs'] = torch.cat([tr_obs['obs'], self.policy_ids_one_hot], dim=-1)
        return tr_obs, rewards, dones, infos

    def inner_play_steps(self):
        update_list = self.update_list
        step_time = 0.0

        if self.is_rnn:
            mb_rnn_states = self.mb_rnn_states
            rnn_state_buffer = [torch.zeros((self.horizon_length, *s.shape), dtype=s.dtype, device=s.device) for s in self.rnn_states]

        for n in range(self.horizon_length):
            if self.is_rnn:
                if n % self.seq_length == 0:
                    for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                        mb_s[n // self.seq_length,:,:,:] = s
                for i, s in enumerate(self.rnn_states):
                    rnn_state_buffer[i][n,:,:,:] = s
                if self.has_central_value:
                    self.central_value_net.pre_step_rnn(n)
            
            res_dict = self.get_action_values(self.obs)
            
            if self.is_rnn:
                self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones.byte())

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            with torch.no_grad():
                obs_tensor_leader = self.obs['obs'].clone()
                leader_one_hot = torch.zeros((obs_tensor_leader.shape[0], self.num_policies), device=self.ppo_device)
                leader_one_hot[:, 0] = 1.0
                obs_tensor_leader[:, -self.num_policies:] = leader_one_hot
                obs_leader_dict = self.obs.copy()
                obs_leader_dict['obs'] = obs_tensor_leader
                off_policy_value_leader = self.get_values(obs_leader_dict)
                self.experience_buffer.update_data('off_policy_value_leader', n, off_policy_value_leader)

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()
            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            if self.normalize_reward:
                self.reward_normalizer.update_stats(
                    shaped_rewards.squeeze(-1),
                    self.dones,
                    self.all_task_indices
                )
            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            if self.is_rnn and len(all_done_indices) > 0:
                if self.zero_rnn_on_done:
                    for s in self.rnn_states:
                        s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                if self.has_central_value:
                    self.central_value_net.post_step_rnn(all_done_indices)

            # leader's (policy 0) finished episodes for tracking
            leader_env_mask = (self.policy_ids_flat == 0)
            leader_done_indices = env_done_indices[leader_env_mask[env_done_indices]]

            if len(leader_done_indices) > 0:
                indices = leader_done_indices.view(-1, 1)
                self.game_rewards.update(self.current_rewards[indices])
                self.game_shaped_rewards.update(self.current_shaped_rewards[indices])
                self.game_lengths.update(self.current_lengths[indices])

            if len(env_done_indices) > 0 and 'episode' in infos and 'success' in infos['episode']:
                done_envs = env_done_indices.view(-1)
                done_policies = self.policy_ids_flat[done_envs]
                done_success = infos['episode']['success'][done_envs].float()
                for p in range(self.num_policies):
                    m = (done_policies == p)
                    if m.any():
                        self.policy_epoch_success_sum[p] += done_success[m].sum()
                        self.policy_epoch_episode_count[p] += m.sum()

            self.algo_observer.process_infos(infos, env_done_indices)
            not_dones = 1.0 - self.dones.float()

            self.current_rewards *= not_dones.unsqueeze(1)
            self.current_shaped_rewards *= not_dones.unsqueeze(1)
            self.current_lengths *= not_dones

        last_values = self.get_values(self.obs)

        with torch.no_grad():
            obs_leader_final_dict = self.obs.copy()
            obs_tensor_leader_final = self.obs['obs'].clone()
            leader_one_hot = torch.zeros((obs_tensor_leader_final.shape[0], self.num_policies), device=self.ppo_device)
            leader_one_hot[:, 0] = 1.0
            obs_tensor_leader_final[:, -self.num_policies:] = leader_one_hot
            obs_leader_final_dict['obs'] = obs_tensor_leader_final
            last_value_leader = self.get_values(obs_leader_final_dict)

        if self.normalize_reward:
            mb_rewards = self.experience_buffer.tensor_dict['rewards']
            for n in range(self.horizon_length):
                mb_rewards[n] = self.reward_normalizer(
                    mb_rewards[n].squeeze(-1),
                    self.all_task_indices
                ).unsqueeze(-1)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values_acting = self.experience_buffer.tensor_dict['values']
        mb_values_leader = self.experience_buffer.tensor_dict['off_policy_value_leader']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        
        # Calculate GAE returns for each policy's own on-policy data.
        mb_advs_acting = self.discount_values(fdones, last_values, mb_fdones, mb_values_acting, mb_rewards)
        mb_returns_acting = mb_advs_acting + mb_values_acting

        # Calculate 1-step returns for off-policy data from leader's perspective.
        all_values_leader = torch.cat([mb_values_leader, last_value_leader.unsqueeze(0)], dim=0)
        all_dones = torch.cat([mb_fdones, fdones.unsqueeze(0)], dim=0)
        mb_one_step_returns_leader = mb_rewards + self.gamma * all_values_leader[1:] * (1.0 - all_dones[1:].unsqueeze(-1))
        self.experience_buffer.tensor_dict['one_step_returns_leader'] = mb_one_step_returns_leader

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns_acting)
        if self.is_rnn:
            # mb_rnn_states[i]: [n_chunks, layers, total_agents, units] -> [layers, total_seqs, units]
            # ordered (agent-major, chunk-within), matching swap_and_flatten01's sequence order so each
            # state lines up with its sequence's transitions.
            batch_dict['rnn_states'] = [
                s.permute(1, 2, 0, 3).reshape(s.shape[1], -1, s.shape[3]) for s in self.mb_rnn_states
            ]
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict
    
    def play_steps(self):
        batch_dict = self.inner_play_steps()
        
        task_indices = self.vec_env.env.extras["task_indices"]
        arr = task_indices.unsqueeze(0).expand(self.horizon_length, -1)
        s = arr.size()
        batch_dict["task_indices"] = arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:]) 

        # Collect data by policy using the dynamically generated policy IDs
        policy_batch_dicts = []

        if self.is_rnn:
            # sequence s belongs to agent (s // n_chunks); its policy is policy_ids_flat[agent].
            n_chunks = self.horizon_length // self.seq_length
            seq_policy_ids = self.policy_ids_flat.repeat_interleave(n_chunks)

        for p_idx in range(self.num_policies):
            policy_mask = (self.policy_ids_flat == p_idx)
            expanded_mask = policy_mask.unsqueeze(0).expand(self.horizon_length, -1)
            flattened_mask = expanded_mask.transpose(0, 1).reshape(-1)

            policy_data = {k: v[flattened_mask] for k, v in batch_dict.items() if isinstance(v, torch.Tensor) and v.shape[0] == flattened_mask.shape[0]}
            if self.is_rnn:
                # rnn_states is a list of [layers, total_seqs, units] (shape[0] == layers, so it is
                # excluded from the per-transition comprehension above); slice it per sequence here.
                seq_mask = (seq_policy_ids == p_idx)
                policy_data['rnn_states'] = [s[:, seq_mask].contiguous() for s in batch_dict['rnn_states']]
            policy_batch_dicts.append(policy_data)

        return batch_dict, policy_batch_dicts

    def augment_and_combine_batches(self, policy_batch_dicts: list) -> dict:
        final_batches_to_combine = []
        for i in range(self.num_policies):
            batch = policy_batch_dicts[i]
            batch['off_policy_mask'] = torch.zeros(batch['obses'].shape[0], dtype=torch.bool, device=self.ppo_device)
            final_batches_to_combine.append(batch)

        num_followers = self.num_policies - 1
        if num_followers > 0:
            # Match official rl_games SAPG (mixed_expl + lf): pick `off_policy_ratio`
            # follower BLOCKS at random without replacement and include ALL transitions
            # from each. off_policy_ratio == -1 uses every follower.
            ratio = int(self.off_policy_ratio)
            if ratio == -1:
                num_follower_blocks = num_followers
            else:
                num_follower_blocks = min(num_followers, ratio)

            if num_follower_blocks > 0:
                chosen = np.random.choice(
                    range(1, self.num_policies), size=num_follower_blocks, replace=False
                ).tolist()
                if self.multi_gpu:
                    dist.broadcast_object_list(chosen, 0)

                chosen_follower_batches = [policy_batch_dicts[i] for i in chosen]
                pooled = {
                    key: torch.cat([b[key] for b in chosen_follower_batches], dim=0)
                    for key in chosen_follower_batches[0].keys()
                    if key != 'rnn_states' and chosen_follower_batches[0][key] is not None
                }

                sample_size = pooled['obses'].shape[0]
                augmented_batch = dict(pooled)
                augmented_batch['values'] = pooled['off_policy_value_leader']
                augmented_batch['returns'] = pooled['one_step_returns_leader']

                leader_one_hot = torch.zeros((sample_size, self.num_policies), device=self.ppo_device)
                leader_one_hot[:, 0] = 1.0
                augmented_batch['obses'] = augmented_batch['obses'].clone()
                augmented_batch['obses'][:, -self.num_policies:] = leader_one_hot

                augmented_batch['off_policy_mask'] = torch.ones(
                    sample_size, dtype=torch.bool, device=self.ppo_device
                )

                if self.is_rnn:
                    # pool the chosen followers' per-sequence states along the sequence dim (dim=1);
                    # the obs are relabelled to the leader but keep the followers' states (off-policy
                    # relabelling approximation, as in the reference SAPG).
                    n_state = len(chosen_follower_batches[0]['rnn_states'])
                    augmented_batch['rnn_states'] = [
                        torch.cat([b['rnn_states'][i] for b in chosen_follower_batches], dim=1)
                        for i in range(n_state)
                    ]

                final_batches_to_combine.append(augmented_batch)

        all_keys = final_batches_to_combine[0].keys()
        combined_batch_dict = {
            key: torch.cat([b[key] for b in final_batches_to_combine if key in b and b[key] is not None], dim=0)
            for key in all_keys if key != 'rnn_states'
        }
        if self.is_rnn:
            n_state = len(final_batches_to_combine[0]['rnn_states'])
            combined_batch_dict['rnn_states'] = [
                torch.cat([b['rnn_states'][i] for b in final_batches_to_combine if 'rnn_states' in b], dim=1)
                for i in range(n_state)
            ]

        return combined_batch_dict

    def prepare_dataset(self, batch_dict: dict):
        values = batch_dict['values']
        returns = batch_dict['returns']
        task_indices = batch_dict["task_indices"]

        advantages = returns - values

        if self.normalize_value:
            unique_tasks = torch.unique(task_indices)
            for idx in unique_tasks:
                mask = task_indices == idx
                self.value_mean_stds[idx.item()].train()
                values[mask] = self.value_mean_stds[idx.item()](values[mask])
                returns[mask] = self.value_mean_stds[idx.item()](returns[mask])
                self.value_mean_stds[idx.item()].eval()

        advantages = torch.sum(advantages, axis=1) # Flatten to 1D

        if self.normalize_advantage:
            # Normalize per task AND per on/off-policy subset. On-policy advantages are
            # GAE; off-policy advantages are 1-step TD residuals (different variance).
            off_mask = batch_dict['off_policy_mask']
            on_mask = ~off_mask
            for task_id in torch.unique(task_indices):
                task_mask = task_indices == task_id
                for subset_mask in (task_mask & on_mask, task_mask & off_mask):
                    if subset_mask.sum() > 1:
                        a = advantages[subset_mask]
                        advantages[subset_mask] = (a - a.mean()) / (a.std() + 1e-8)
                
        if self.is_rnn:
            # Shuffle at the SEQUENCE granularity: the PPODataset slices contiguous sequence
            # ranges, so a per-transition shuffle would scramble each BPTT window. Permuting whole
            # sequences keeps each window intact (and its rnn_state aligned) while still mixing
            # tasks / on- and off-policy data into the sampled minibatches.
            num_seqs = len(advantages) // self.seq_length
            seq_perm = torch.randperm(num_seqs, device=advantages.device) if self.shuffle_data \
                else torch.arange(num_seqs, device=advantages.device)
            perm = (
                seq_perm.unsqueeze(1) * self.seq_length
                + torch.arange(self.seq_length, device=advantages.device)
            ).view(-1)
        else:
            perm = torch.randperm(len(advantages)) if self.shuffle_data else torch.arange(len(advantages))

        dataset_dict = {
            'advantages': advantages[perm],
            'old_values': values.squeeze(-1)[perm],
            'returns': returns.squeeze(-1)[perm],
            'obs': batch_dict['obses'][perm],
            'actions': batch_dict['actions'][perm],
            'old_logp_actions': batch_dict['neglogpacs'][perm],
            'mu': batch_dict['mus'][perm],
            'sigma': batch_dict['sigmas'][perm],
            'task_indices': batch_dict['task_indices'][perm],
            'off_policy_mask': batch_dict['off_policy_mask'][perm],
        }
        if self.is_rnn:
            dataset_dict['dones'] = batch_dict['dones'][perm]
            # full sequences (no padding) -> all-ones mask; calc_gradients reads this for kl averaging.
            dataset_dict['rnn_masks'] = torch.ones(len(advantages), device=advantages.device)
            dataset_dict['rnn_states'] = [s[:, seq_perm].contiguous() for s in batch_dict['rnn_states']]

        self.dataset.update_values_dict(dataset_dict)
        self.dataset.values_dict["task_indices"] = batch_dict["task_indices"][perm]

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
            'task_indices': task_indices,
            'rnn_states': input_dict.get('rnn_states', None)
        }
        if self.is_rnn:
            batch_dict['dones'] = input_dict['dones']
            batch_dict['seq_length'] = self.seq_length

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values'].squeeze(-1)
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(
                old_action_log_probs_batch, 
                action_log_probs, 
                advantage, 
                self.ppo, 
                curr_e_clip, 
                task_indices
            )

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
            
            ec_candidates = self.policy_coeffs
            ec_indices = torch.argmax((obs_batch[:,-self.num_policies:]), dim=1)
            entropy_coef = ec_candidates[ec_indices]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * entropy_coef + b_loss * self.bounds_loss_coef

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
        
        self.scaler.scale(loss.mean()).backward()

        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        return a_loss, c_loss, entropy, kl_dist, self.last_lr, lr_mul, mu.detach(), sigma.detach(), b_loss

    def train_actor_critic(self, input_dict):
        return self.calc_gradients(input_dict)

    def _log_per_policy_value_stats(self, batch_dicts_by_block):
        # Diagnostic for the mixed-policy value normalizer question: if per-policy V
        # scales stay within ~2x of each other across training, the single-(per-task)
        # value_mean_std is fine; if they diverge by 5x+ (especially early), consider
        # splitting value_mean_stds by [task, policy].
        if self.writer is None:
            return
        with torch.no_grad():
            means, stds = [], []
            for p in range(self.num_policies):
                v = batch_dicts_by_block[p]['values'].float()
                m = v.mean().item()
                s = v.std().item()
                means.append(m)
                stds.append(s)
                self.writer.add_scalar(f'sapg/value_mean_policy_{p}', m, self.frame)
                self.writer.add_scalar(f'sapg/value_std_policy_{p}', s, self.frame)
            if self.num_policies > 1:
                abs_means = [abs(x) for x in means]
                self.writer.add_scalar('sapg/value_mean_spread', max(abs_means) - min(abs_means), self.frame)
                self.writer.add_scalar('sapg/value_std_spread', max(stds) - min(stds), self.frame)

    def _log_and_reset_per_policy_success(self):
        # Per-epoch exact success rate: sum(successes this epoch) / sum(episodes this
        # epoch), per policy and overall. Hparam-invariant — no sliding window, no
        # per-step weighting bias.
        total_succ = self.policy_epoch_success_sum.sum().item()
        total_eps = self.policy_epoch_episode_count.sum().item()
        if self.writer is not None and total_eps > 0:
            self.writer.add_scalar('sapg/success_rate_all', total_succ / total_eps, self.frame)
            for p in range(self.num_policies):
                n = self.policy_epoch_episode_count[p].item()
                if n > 0:
                    rate = (self.policy_epoch_success_sum[p] / self.policy_epoch_episode_count[p]).item()
                    self.writer.add_scalar(f'sapg/success_rate_policy_{p}', rate, self.frame)
        self.policy_epoch_success_sum.zero_()
        self.policy_epoch_episode_count.zero_()

    def _log_per_policy_consecutive_successes(self):
        """Leader vs follower (and per-task) mean *consecutive* successes.

        The binary infos['episode']['success'] that success_rate_policy_* relied on is
        intentionally absent for AllegroKuka, so we snapshot the env's prev_episode_successes
        (same quantity as the global successes/<task> metric) and slice it by policy block.
        Policy 0 is the SAPG leader; policies 1.. are followers.
        """
        if self.writer is None:
            return
        env = self.vec_env.env
        if "successes" not in env.extras:
            return
        succ = env.extras["successes"].to(self.ppo_device).float()  # [num_actors], prev-episode consecutive successes
        task_names = env.extras.get("ordered_task_names", None)
        leader_mask = (self.policy_ids_flat == 0)
        groups = {'leader': leader_mask, 'followers': ~leader_mask}

        for gname, gmask in groups.items():
            if gmask.any():
                self.writer.add_scalar(f'sapg/successes_{gname}', succ[gmask].mean().item(), self.frame)
            for t in range(self.task_embedding_dim):
                tmask = gmask & (self.all_task_indices == t)
                if tmask.any():
                    tname = task_names[t] if task_names is not None and t < len(task_names) else str(t)
                    self.writer.add_scalar(f'sapg/successes_{gname}/{tname}', succ[tmask].mean().item(), self.frame)

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            batch_dict, batch_dicts_by_block = self.play_steps()

        self._log_per_policy_value_stats(batch_dicts_by_block)
        self._log_and_reset_per_policy_success()
        self._log_per_policy_consecutive_successes()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        
        final_batch_dict = self.augment_and_combine_batches(batch_dicts_by_block)
        self.prepare_dataset(final_batch_dict)

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

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, self.last_lr, 1.0