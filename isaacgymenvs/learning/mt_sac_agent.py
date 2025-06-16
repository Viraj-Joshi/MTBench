from typing import List, Tuple

import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
from scipy.optimize import minimize

import torch 
from torch import nn
from torch import optim

from rl_games.algos_torch.sac_agent import SACAgent
from rl_games.algos_torch import torch_ext

from isaacgymenvs.learning.replay.nstep_replay import NStepReplay
from isaacgymenvs.learning.replay.simple_replay import ReplayBuffer


class MTSACAgent(SACAgent):
    def __init__(self, base_name, params):
        self.config = config = params['config']
        print(config)

        # TODO: Get obs shape and self.network
        self.load_networks(params)
        self.base_init(base_name, config)
        self.num_warmup_steps = config["num_warmup_steps"]
        self.gamma = config["gamma"]
        self.critic_tau = float(config["critic_tau"])
        self.batch_size = config["batch_size"]
        self.init_alpha = config["init_alpha"]
        self.learnable_temperature = config["learnable_temperature"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.horizon = config["horizon"]
        self.normalize_input = config.get("normalize_input", False)
        self.normalize_value = config.get("normalize_value", False)
        self.gradient_steps_per_itr = config["gradient_steps_per_itr"]
        self.grad_norm = config["grad_norm"]
        self.nstep = config.get("nstep", 1)
        self.actor_update_freq = config.get("actor_update_freq", 1)
        self.critic_target_update_freq = config.get("critic_target_update_freq", 1)

        self.use_replay_ratio_scaling = config.get("use_replay_ratio_scaling", False)
        self.replay_ratio_scaling_update_freq = config.get("replay_ratio_scaling_update_freq", None)

        # TODO: double-check! To use bootstrap instead?
        self.max_env_steps = config.get("max_env_steps", 1000) # temporary, in future we will use other approach

        print(self.batch_size, self.num_actors, self.num_agents)

        self.num_frames_per_epoch = self.num_actors * self.horizon
        self.num_updates = 0

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).float().to(self._device)
        self.log_alpha.requires_grad = True
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)

        self.all_task_indices : torch.Tensor = self.vec_env.env.extras["task_indices"]
        ordered_task_names : list[str] = self.vec_env.env.extras["ordered_task_names"]

        self.task_embedding_dim = torch.unique(self.all_task_indices).shape[0]
        
        net_config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'normalize_input': self.normalize_input,
            'normalize_value': self.normalize_value,
            'task_indices': self.all_task_indices,
            'task_embedding_dim': self.task_embedding_dim,
        }
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)

        self.actor_optimizer = torch.optim.Adam(self.model.sac_network.actor.parameters(),
                                                lr=float(self.config['actor_lr']),
                                                betas=self.config.get("actor_betas", [0.9, 0.999]))

        self.critic_optimizer = torch.optim.Adam(self.model.sac_network.critic.parameters(),
                                                 lr=float(self.config["critic_lr"]),
                                                 betas=self.config.get("critic_betas", [0.9, 0.999]))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=float(self.config["alpha_lr"]),
                                                    betas=self.config.get("alphas_betas", [0.9, 0.999]))

        self.target_entropy_coef = config.get("target_entropy_coef", 1.0)
        self.target_entropy = self.target_entropy_coef * -self.env_info['action_space'].shape[0]
        print("Target entropy", self.target_entropy)

        self.algo_observer = config['features']['observer']

        obs_dim = self.env_info["observation_space"].shape[0]
        action_dim = self.env_info["action_space"].shape[0]
        self.n_step_buffer = NStepReplay(obs_dim,
                                         action_dim,
                                         self.num_actors,
                                         self.nstep,
                                         device=self.device)    
        
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, obs_dim, action_dim, self.device)

        self.algo_observer.before_init(base_name, self.config, self.experiment_name)

        if self.config.get("use_disentangled_alpha", False):
            NUM_TASKs = torch.unique(self.all_task_indices).shape[0]
            self.log_alpha = nn.Parameter(torch.tensor(np.log([self.init_alpha] * NUM_TASKs), requires_grad=True).float().to(self._device))
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=float(self.config["alpha_lr"]),
                                                    betas=self.config.get("alphas_betas", [0.9, 0.999]))

    # @property
    def alpha(self, task_indices: torch.Tensor, detach: bool = True):
        """
        Retreive the temperature parameter alpha for the given task indices.

        Parameters:
        task_indices (torch.Tensor): Indices of the tasks.
        detach (bool): Whether to detach the result from the computation graph.

        Returns:
        torch.Tensor: The temperature parameter alpha for the given task indices.
        """
        if self.config.get("use_disentangled_alpha", False):
            if detach:
                return torch.exp(self.log_alpha[task_indices]).unsqueeze(-1).detach()
            return torch.exp(self.log_alpha[task_indices]).unsqueeze(-1) # (B,1)
        else:
            if detach:
                return torch.exp(self.log_alpha).unsqueeze(0).repeat(task_indices.shape[0], 1).detach()
            else:
                return torch.exp(self.log_alpha).unsqueeze(0).repeat(task_indices.shape[0], 1) # (B,1)

    def set_weights(self, weights):
        self.model.sac_network.actor.load_state_dict(weights['actor'])
        self.model.sac_network.critic.load_state_dict(weights['critic'])
        self.model.sac_network.critic_target.load_state_dict(weights['critic_target'])

        if self.normalize_input and 'running_mean_stds' in weights:
            for k in range(50):
                self.model.running_mean_stds[k].load_state_dict(weights['running_mean_stds'][k])

    def preproc_obs(self, obs, task_indices):
        if isinstance(obs, dict):
            obs = obs['obs']
        obs = self.model.norm_obs(obs, task_indices)
        return obs

    def act(self, obs, action_dim, task_indices, sample=False):
        obs = self.preproc_obs(obs, task_indices)
        dist = self.model.actor(obs)

        actions = dist.sample() if sample else dist.mean
        actions = actions.clamp(*self.action_range)
        assert actions.ndim == 2

        return actions
    
    def train_epoch(self):
        if self.use_replay_ratio_scaling and self.num_updates!=0 and self.num_updates % self.replay_ratio_scaling_update_freq == 0:
            print(f"resetting model parameters at {self.num_updates} updates")
            self.model.reset_all_parameters()
        
        if self.epoch_num == 1:
            return self.play_steps(self.num_warmup_steps, True)
        else:
            return self.play_steps(self.horizon, False) 
        
    def play_steps(self, horizon, random_exploration):
        total_time_start = time.time()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        actor_losses = []
        entropies = []
        alphas = []
        alpha_losses = []
        critic1_loss_list = []
        critic2_loss_list = []

        obs_dim = self.env_info["observation_space"].shape[0]
        action_dim = self.env_info["action_space"].shape[0]
        traj_obs = torch.empty((self.num_actors, horizon) + (obs_dim,), device=self.device)
        traj_actions = torch.empty((self.num_actors, horizon) + (action_dim,), device=self.device)
        traj_rewards = torch.empty((self.num_actors, horizon), device=self.device)
        traj_next_obs = torch.empty((self.num_actors, horizon) + (obs_dim,), device=self.device)
        traj_dones = torch.empty((self.num_actors, horizon), device=self.device)

        obs = self.obs
        if isinstance(obs, dict):
            obs = self.obs['obs']

        next_obs_processed = obs.clone()

        for s in range(horizon):
            self.set_eval()
            if random_exploration:
                print(f"Warmup Step: {s}")
                action = torch.rand((self.num_actors, *self.env_info["action_space"].shape), device=self._device) * 2.0 - 1.0
            else:
                with torch.no_grad():
                    action = self.act(obs.float(), self.env_info["action_space"].shape, self.all_task_indices, sample=True)

            step_start = time.time()
            with torch.no_grad():
                next_obs, rewards, dones, infos = self.env_step(action)
            step_end = time.time()

            self.current_rewards += rewards
            self.current_lengths += 1

            total_time += (step_end - step_start)
            step_time += (step_end - step_start)

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - dones.float()

            self.algo_observer.process_infos(infos, done_indices)

            no_timeouts = self.current_lengths != self.max_env_steps
            dones = dones * no_timeouts

            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

            if isinstance(next_obs, dict):    
                next_obs_processed = next_obs['obs']

            self.obs = next_obs.copy() # changed from .clone()
            rewards = self.rewards_shaper(rewards)
            if (torch.max(obs[:,-self.task_embedding_dim:], dim=1)[0]==0).any():
                obs[:,-self.task_embedding_dim:] = torch.nn.functional.one_hot(self.all_task_indices,self.task_embedding_dim)
                print(f"obs was corrupted at step {s} of epoch {self.epoch_num} in play steps after stepping environment. Fixed it")
            if (torch.max(next_obs_processed[:,-self.task_embedding_dim:], dim=1)[0]==0).any():
                next_obs_processed[:,-self.task_embedding_dim:] = torch.nn.functional.one_hot(self.all_task_indices,self.task_embedding_dim)
                print(f"next obs was corrupted at step {s} of epoch {self.epoch_num} in play steps after stepping environment. Fixed it.")

            if isinstance(obs, dict):
                obs = self.obs['obs']

            # record transitions
            traj_obs[:, s] = obs
            traj_actions[:, s] = action
            traj_dones[:, s] = dones
            traj_rewards[:, s] = rewards
            traj_next_obs[:, s] = next_obs_processed

        traj_rewards = traj_rewards.reshape(self.num_actors,horizon, 1)
        traj_dones = traj_dones.reshape(self.num_actors, horizon, 1)
        data = self.n_step_buffer.add_to_buffer(traj_obs, traj_actions, traj_rewards, traj_next_obs, traj_dones)

        self.replay_buffer.add_to_buffer(data)


        if not random_exploration:
            self.set_train()
            update_time_start = time.time()
            for gradient_step in range(self.gradient_steps_per_itr):
                actor_loss_info, critic1_loss, critic2_loss = self.update(self.epoch_num)
            self.num_updates += self.gradient_steps_per_itr
            update_time_end = time.time()
            update_time = update_time_end - update_time_start

            self.extract_actor_stats(actor_losses, entropies, alphas, alpha_losses, actor_loss_info)
            critic1_loss_list.append(critic1_loss)
            critic2_loss_list.append(critic2_loss)
        else:
            update_time = 0

        total_update_time += update_time

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        return step_time, play_time, total_update_time, total_time, actor_losses, entropies, alphas, alpha_losses, critic1_loss_list, critic2_loss_list

    def get_actions_logprob(self, state: torch.Tensor):
        dist = self.model.actor(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, dist, log_prob
    
    def update_critic(self, obs, action, reward, next_obs, not_done, step, task_indices):
        with torch.no_grad():
            next_action, dist, log_prob = self.get_actions_logprob(next_obs)
            # (B,4), (B,), (B,1)
            target_Q1, target_Q2 = self.model.critic_target(next_obs, next_action)
            # (B,1), (B,1)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha(task_indices) * log_prob
            # (B,1) - (B,1) * (B,1) -> (B,1)
            target_Q = reward + (not_done) * (self.gamma ** self.nstep) * target_V
            # (B,1)

        # get current Q estimates
        current_Q1, current_Q2 = self.model.critic(obs, action)
        # (B,1), (B,1)  
        critic1_loss = self.c_loss(current_Q1, target_Q)
        critic2_loss = self.c_loss(current_Q2, target_Q)
        critic_loss = critic1_loss + critic2_loss 
        self.critic_optimizer.zero_grad(set_to_none=True)
        nn.utils.clip_grad_norm_(self.model.sac_network.critic.parameters(), max_norm=self.grad_norm)
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.detach(), critic1_loss.detach(), critic2_loss.detach()

    def update(self, step):
        obs, action, reward, next_obs, done, _ = self.replay_buffer.sample_batch(self.batch_size)
        not_done = 1-done

        task_indices = torch.argmax(obs[:,-self.task_embedding_dim:], dim=1)

        obs = self.preproc_obs(obs, task_indices)
        next_obs = self.preproc_obs(next_obs, task_indices)

        critic_loss, critic1_loss, critic2_loss = self.update_critic(obs, action, reward, next_obs, not_done, step, task_indices)

        actor_loss, entropy, alpha, alpha_loss = self.update_actor_and_alpha(obs, step, task_indices)

        actor_loss_info = actor_loss, entropy, alpha, alpha_loss
        self.soft_update_params(self.model.sac_network.critic, self.model.sac_network.critic_target,
                                     self.critic_tau)
        return actor_loss_info, critic1_loss, critic2_loss

    def update_actor_and_alpha(self, obs, step, task_indices):
        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = False

        action, dist, log_prob = self.get_actions_logprob(obs)
        # (B,4), (B,), (B,1)
        entropy = -log_prob.mean()
        actor_Q1, actor_Q2 = self.model.critic(obs, action)
        # (B,1), (B,1)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        # (B,1)

        actor_loss = (self.alpha(task_indices) * log_prob - actor_Q)
        # (B,1) * (B,1) - (B,1) -> (B,1)
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        nn.utils.clip_grad_norm_(self.model.sac_network.actor.parameters(), max_norm=self.grad_norm)
        actor_loss.backward()

        self.actor_optimizer.step()

        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = True

        if self.learnable_temperature:
            alpha_loss = (
                self.alpha(task_indices,detach=False)
                * (-log_prob - self.target_entropy).detach()
            ).mean()
            
            self.log_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            nn.utils.clip_grad_norm_(self.log_alpha, max_norm=self.grad_norm)
            self.log_alpha_optimizer.step()
        else:
            alpha_loss = None

        return actor_loss.detach(), entropy.detach(), self.alpha(task_indices).mean(), alpha_loss # TODO: maybe not self.alpha

# balance task losses as described in the Soft Modularization paper
class MTSACSoftModularizationAgent(MTSACAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        self.c_loss = nn.MSELoss(reduction='none')
        self.encoder_tau = self.config["encoder_tau"]

        self.unique_tasks = torch.unique(self.all_task_indices)

    def update_critic(self, obs, action, reward, next_obs, not_done, step, task_indices):
        """
        Update critic networks for multi-task SAC with automatic task weighting.
        
        Implements Eq. 10 for task weighting and applies it to the Q-function objective (Eq. 4)
        from the paper. Tasks with higher uncertainty (higher temperature α) get higher weights.
        """
        with torch.no_grad():
            next_action, dist, log_prob = self.get_actions_logprob(next_obs)
            # (B,4), (B,), (B,1)
            target_Q1, target_Q2 = self.model.critic_target(next_obs, next_action)
            # (B,1), (B,1)
            alphas = self.alpha(task_indices)
            # (B,1)
            target_V = torch.min(target_Q1, target_Q2) - alphas * log_prob
            # (B,1) - (B,1) * (B,1) -> (B,1)
            target_Q = reward + (not_done) * (self.gamma ** self.nstep) * target_V
            # (B,1)

            
        
        # get current Q estimates
        current_Q1, current_Q2 = self.model.critic(obs, action)
        # (B,1), (B,1)
        critic1_losses = self.c_loss(current_Q1, target_Q)
        # (B,1)
        critic2_losses = self.c_loss(current_Q2, target_Q)
        # (B,1)

        # calculate the task weights for ALL tasks being trained, not just the tasks sampled in batch
        alphas = self.alpha(self.unique_tasks)
        # (len(unique_tasks),1)
        task_weights = torch.nn.functional.softmax(-alphas,dim=0).detach() # Eq.10 gives you a weight for each task, not each environment
        # (len(unique_tasks),1)

        if torch.isnan(task_weights).any():
            raise RuntimeError("NaN detected in task weights!")

        # Compute weighted losses for each critic network
        batch_weights = task_weights[task_indices]
        # (B,1)
        
        weighted_critic1_loss = (batch_weights * critic1_losses).mean()
        weighted_critic2_loss = (batch_weights * critic2_losses).mean()
        critic_loss = weighted_critic1_loss + weighted_critic2_loss

        if (critic_loss > 2e8).any():
            raise RuntimeError("Critic loss has diverged!")

        # Update critic networks
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.sac_network.critic.parameters(), 
            max_norm=self.grad_norm
        )
        self.critic_optimizer.step()

        return (
            critic_loss.detach(),
            weighted_critic1_loss.detach(),
            weighted_critic2_loss.detach()
        )
    
    def update_actor_and_alpha(self, obs, step, task_indices):
        """
        Update actor network and temperature parameter alpha for multi-task SAC.
        
        Implements weighted actor loss using task weights w_i from Eq. 10 and updates
        the temperature parameter if learnable_temperature is enabled.
        """
        # Temporarily disable critic gradients during actor update
        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = False

        action, _, log_prob = self.get_actions_logprob(obs)
        # (B,4), (B,), (B,1)
        entropy = -log_prob.mean()
        actor_Q1, actor_Q2 = self.model.critic(obs, action)
        # (B,1), (B,1)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        # (B,1)
        # calculate the task weights for ALL tasks being trained, not just the tasks sampled in batch
        alphas = self.alpha(self.unique_tasks)
        # (len(unique_tasks),1)
        task_weights = torch.nn.functional.softmax(-alphas,dim=0).detach() # Eq.10 gives you a weight for each task, not each environment
        # (len(unique_tasks),1)

        # get the task weights for each environment in the batch
        batch_weights = task_weights[task_indices]
        # (B,1)

        # Compute weighted actor loss: E_τ[w_τ * (α * log_prob - Q)]
        actor_loss = batch_weights * (self.alpha(task_indices) * log_prob - actor_Q)
        # (B,1) * (B,1) - (B,1) -> (B,1)
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.model.sac_network.actor.parameters(), max_norm=self.grad_norm)
        self.actor_optimizer.step()

        # Re-enable critic gradients
        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = True

        if self.learnable_temperature:
            # α_loss = E[α * (-log_prob - target_entropy)]
            alpha_loss = (
                self.alpha(task_indices,detach=False)
                * (-log_prob - self.target_entropy).detach()
            ).mean()

            self.log_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            nn.utils.clip_grad_norm_(self.log_alpha, max_norm=self.grad_norm)
            self.log_alpha_optimizer.step()
        else:
            alpha_loss = None

        return actor_loss.detach(), entropy.detach(), self.alpha(task_indices).mean().detach(), alpha_loss
    
    def update(self, step):
        obs, action, reward, next_obs, done, _ = self.replay_buffer.sample_batch(self.batch_size)
        task_indices = torch.argmax(obs[:,-self.task_embedding_dim:], dim=1)
        not_done = 1-done

        obs = self.preproc_obs(obs, task_indices)
        next_obs = self.preproc_obs(next_obs, task_indices)
        critic_loss, critic1_loss, critic2_loss = self.update_critic(obs, action, reward, next_obs, not_done, step, task_indices)

        actor_loss, entropy, alpha, alpha_loss = self.update_actor_and_alpha(obs, step, task_indices)

        actor_loss_info = actor_loss, entropy, alpha, alpha_loss

        # update target networks
        self.soft_update_params(self.model.sac_network.critic.Q1, self.model.sac_network.critic_target.Q1,
                                     self.critic_tau)
        self.soft_update_params(self.model.sac_network.critic.Q2, self.model.sac_network.critic_target.Q2,
                                     self.critic_tau)
        
        self.soft_update_params(self.model.sac_network.critic.state_action_encoder_1, self.model.sac_network.critic_target.state_action_encoder_1,
                                     self.encoder_tau)
        self.soft_update_params(self.model.sac_network.critic.state_action_encoder_2, self.model.sac_network.critic_target.state_action_encoder_2,
                                     self.encoder_tau)
        
        self.soft_update_params(self.model.sac_network.critic.task_encoder_1, self.model.sac_network.critic_target.task_encoder_1,
                                     self.encoder_tau)
        self.soft_update_params(self.model.sac_network.critic.task_encoder_2, self.model.sac_network.critic_target.task_encoder_2,
                                        self.encoder_tau)

        return actor_loss_info, critic1_loss, critic2_loss