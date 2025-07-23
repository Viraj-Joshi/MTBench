import numpy as np
import time
from datetime import datetime
import os

import torch 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from rl_games.algos_torch import torch_ext
from rl_games.common import vecenv
from rl_games.common.a2c_common import print_statistics
from rl_games.interfaces.base_algorithm import BaseAlgorithm
from rl_games.algos_torch import  model_builder

from isaacgymenvs.learning.mt_models import PerTaskRewardNormalizer
from isaacgymenvs.learning.replay.nstep_replay import NStepReplay
from isaacgymenvs.learning.replay.simple_replay import ReplayBuffer

import wandb

import matplotlib.pyplot as plt
class FastTD3Agent(BaseAlgorithm):

    def __init__(self, base_name, params):

        self.config = config = params['config']
        print(config)

        self.load_networks(params)
        self.base_init(base_name, config)
        self.num_warmup_steps = config["num_warmup_steps"]
        self.gamma = config["gamma"]
        self.critic_tau = float(config["critic_tau"])
        self.batch_size = config["batch_size"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.horizon = config["horizon"]
        self.normalize_input = config.get("normalize_input", False)
        self.gradient_steps_per_itr = config["gradient_steps_per_itr"]
        self.use_grad_norm = config['use_grad_norm']
        self.grad_norm = config.get('grad_norm', 0.5)
        self.noise_clip = config["noise_clip"]
        self.policy_noise = config["policy_noise"]
        self.actor_update_freq = config["actor_update_freq"]
        self.nstep = config.get("nstep", 1)
        self.disable_bootstrap = config.get("disable_bootstrap", False)
        amp = config.get("amp", False)
        amp_dtype = config.get("amp_dtype", "bf16")
        self.use_cdq = config.get["use_cdq"]
        self.eval_interval = config.get("eval_interval", 1000)

        self.amp_enabled = amp and torch.cuda.is_available()
        self.amp_device_type = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

        self.scaler = GradScaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)

        # TODO: double-check! To use bootstrap instead?
        self.max_env_steps = config.get("max_env_steps", 1000) # temporary, in future we will use other approach

        print(self.batch_size, self.num_actors, self.num_agents)

        self.num_frames_per_epoch = self.num_actors * self.horizon

        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        net_config = {
            'action_shape' : self.env_info["action_space"].shape,
            'input_shape' : obs_shape,
            'num_envs': self.num_actors,
            'device': self.device,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)

        self.actor_optimizer = torch.optim.AdamW(self.model.td3_network.actor.parameters(),
                                                lr=float(self.config['actor_lr']),
                                                betas=self.config.get("actor_betas", [0.9, 0.999]),
                                                weight_decay=self.config.get("weight_decay", 0.0))

        self.critic_optimizer = torch.optim.AdamW(self.model.td3_network.critic.parameters(),
                                                 lr=float(self.config["critic_lr"]),
                                                 betas=self.config.get("critic_betas", [0.9, 0.999]),
                                                 weight_decay=self.config.get("weight_decay", 0.0))
        
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer,
            T_max=self.max_epochs,
            eta_min=float(self.config['critic_learning_rate_end'])
        )
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            T_max=self.max_epochs,
            eta_min=float(self.config['actor_learning_rate_end'])
        )
        
        
        obs_dim = self.env_info["observation_space"].shape[0]
        action_dim = self.env_info["action_space"].shape[0]
        self.n_step_buffer = NStepReplay(obs_dim,
                                         action_dim,
                                         self.num_actors,
                                         self.nstep,
                                         device=self.device)    
        
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, obs_dim, action_dim, self.device)

        self.algo_observer = config['features']['observer']

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def base_init(self, base_name, config):
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name = config['env_name']
        print("Env name:", self.env_name)

        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self._device = config.get('device', 'cuda:0')

        #temporary for Isaac gym compatibility
        self.ppo_device = self._device
        print('Env info:')
        print(self.env_info)

        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        #self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.save_best_after = config.get('save_best_after', 500)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.save_freq = config.get('save_frequency', 0)

        self.network = config['network']
        self.num_agents = self.env_info.get('agents', 1)
        self.obs_shape = self.observation_space.shape

        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.obs = None

        self.frame = 0
        self.epoch_num = 0
        self.update_time = 0
        self.last_mean_rewards = -1000000000
        self.play_time = 0

        # TODO: put it into the separate class
        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.writer = SummaryWriter('runs/' + config['name'] + datetime.now().strftime("_%d-%H-%M-%S"))
        print("Run Directory:", config['name'] + datetime.now().strftime("_%d-%H-%M-%S"))

        self.is_tensor_obses = False
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None

    def init_tensors(self):
        if self.observation_space.dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            torch_dtype = torch.float32
        batch_size = self.num_agents * self.num_actors

        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)

        self.dones = None

    @property
    def device(self):
        return self._device

    def get_weights(self):
        print("Loading weights")
        state = {'actor': self.model.td3_network.actor.state_dict(),
         'critic': self.model.td3_network.critic.state_dict(), 
         'critic_target': self.model.td3_network.critic_target.state_dict()}
        return state

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def set_weights(self, weights):
        self.model.td3_network.actor.load_state_dict(weights['actor'])
        self.model.td3_network.critic.load_state_dict(weights['critic'])
        self.model.td3_network.critic_target.load_state_dict(weights['critic_target'])

        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])

    def get_full_state_weights(self):
        print("Loading full weights")
        state = self.get_weights()

        state['epoch'] = self.epoch_num
        state['frame'] = self.frame
        state['actor_optimizer'] = self.actor_optimizer.state_dict()
        state['critic_optimizer'] = self.critic_optimizer.state_dict()      

        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)

        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']

        self.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        self.critic_optimizer.load_state_dict(weights['critic_optimizer'])

        self.last_mean_rewards = weights.get('last_mean_rewards', -1000000000)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

    def restore(self, fn, set_epoch=True):
        print("TD3 restore")
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def get_param(self, param_name):
        pass

    def set_param(self, param_name, param_value):
        pass

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    def update_critic(self, obs, action, reward, next_obs, not_done, effective_n_steps):
        with autocast(
            device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled
        ):
            # compute target actions with noise
            clipped_noise = torch.randn_like(action)
            clipped_noise = clipped_noise.mul(self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.model.actor(next_obs) + clipped_noise).clamp(
                self.action_range[0], self.action_range[1]
            )

            if self.disable_bootstrap:
                bootstrap = not_done.float()
            else:
                bootstrap = not_done.float()
                # bootstrap = (truncations | ~dones).float()

            discount = self.gamma ** effective_n_steps
            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = (
                    self.model.td3_network.critic_target.projection(
                        next_obs,
                        next_action,
                        reward.squeeze(-1),
                        bootstrap.squeeze(-1),
                        discount.squeeze(-1),
                    )
                )
                qf1_next_target_value = self.model.td3_network.critic_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = self.model.td3_network.critic_target.get_value(qf2_next_target_projected)

                if self.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist = qf1_next_target_projected
                    qf2_next_target_dist = qf2_next_target_projected

            qf1, qf2 = self.model.critic(obs, action)
            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(qf_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)

        if self.use_grad_norm:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.td3_network.critic.parameters(),
                max_norm=self.grad_norm if self.grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self._device)

        self.scaler.step(self.critic_optimizer)
        self.scaler.update()

        return qf_loss.detach(), qf1_loss.detach(), qf2_loss.detach(), critic_grad_norm.detach()

    def update_actor(self, obs):
        with autocast(
            device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled
        ): 
            action = self.model.actor(obs)
            qf1, qf2 =  self.model.td3_network.critic(obs, action)
            qf1_value = self.model.td3_network.critic.get_value(F.softmax(qf1, dim=1))
            qf2_value = self.model.td3_network.critic.get_value(F.softmax(qf2, dim=1))
            if self.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        if self.use_grad_norm:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.td3_network.actor.parameters(),
                max_norm=self.grad_norm if self.grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self._device)
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()

        return actor_loss.detach(), actor_grad_norm.detach()

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1.0 - tau) * target_param.data)

    def update(self, epoch_num, gradient_step):
        obs, action, raw_reward, next_obs, done, effective_n_steps = self.replay_buffer.sample_batch(self.batch_size)
        if self.normalize_reward:
            task_ids_one_hot = obs[...,-self.num_tasks:]
            task_indices = torch.argmax(task_ids_one_hot, dim=1)
            reward = self.reward_normalizer(raw_reward.squeeze(-1), task_indices)
        not_done = ~(done.bool())

        # obs = self.obs_normalizer(obs)
        # next_obs = self.obs_normalizer(next_obs)
        obs = self.preproc_obs(obs)
        next_obs = self.preproc_obs(next_obs)
        critic_loss, critic1_loss, critic2_loss, critic_grad_norm = self.update_critic(obs, action, reward, next_obs, not_done, effective_n_steps)
        
        # with torch.no_grad():
        #     if epoch_num % 10 == 0 and gradient_step == self.gradient_steps_per_itr - 1:
        #         q1_logits, q2_logits = self.model.td3_network.critic(obs, action)
        #         q1_probs = F.softmax(q1_logits, dim=1).detach().cpu().numpy()
        #         q2_probs = F.softmax(q2_logits, dim=1).detach().cpu().numpy()
        #         # Get the support (the x-axis values)
        #         support = self.model.td3_network.critic.q_support.cpu().numpy()

        #         # 5. Plot the distributions
        #         plt.figure(figsize=(14, 7))
        #         plt.title('Critic Output Distribution for a Single Sample')

        #         # We access the first element of the batch with index [0]
        #         plt.bar(support, q1_probs[0], alpha=0.7, label='QNet1 Distribution')
        #         plt.bar(support, q2_probs[0], alpha=0.5, label='QNet2 Distribution')

        #         plt.xlabel('Q-value')
        #         plt.ylabel('Probability')
        #         plt.legend()
        #         plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        #         plt.show()

        #         plt.savefig(f'debug/critic_distribution_single_sample_{epoch_num}.png')
        #         plt.close()
                
        if gradient_step % self.actor_update_freq == 1:
            actor_loss, actor_grad_norm = self.update_actor(obs)
        else:
            actor_loss = None
            actor_grad_norm = None

        actor_loss_info = actor_loss
        self.soft_update_params(self.model.td3_network.critic, self.model.td3_network.critic_target,
                                     self.critic_tau)
        return actor_loss_info, critic1_loss, critic2_loss, critic_grad_norm, actor_grad_norm

    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs['obs']
        obs = self.model.norm_obs(obs)

        return obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(self.observation_space.dtype != np.int8)
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self._device)
            else:
                obs = torch.FloatTensor(obs).to(self._device)

        return obs

    # TODO: move to common utils
    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}

        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)

        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions) # (obs_space) -> (n, obs_space)

        if self.normalize_reward:
            self.reward_normalizer.update_stats(rewards.squeeze(-1), dones, self.all_task_indices)

        if self.is_tensor_obses:
            return self.obs_to_tensors(obs), rewards.to(self._device), dones.to(self._device), infos
        else:
            return torch.from_numpy(obs).to(self._device), torch.from_numpy(rewards).to(self._device), torch.from_numpy(dones).to(self._device), infos

    def env_reset(self):
        with torch.no_grad():
            obs = self.vec_env.reset()

        obs = self.obs_to_tensors(obs)

        return obs

    def act(self, obs):
        # obs = self.obs_normalizer(obs)
        obs = self.preproc_obs(obs)
        action = self.model.td3_network.actor.explore(obs,self.dones)

        action = action.clamp(*self.action_range)
        assert action.ndim == 2

        return action

    def extract_actor_stats(self, actor_losses, actor_loss_info):
        actor_loss = actor_loss_info

        actor_losses.append(actor_loss)

    def clear_stats(self):
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.algo_observer.after_clear_stats()

    def evaluate(self):
        """
        Evaluates the actor on using a deterministic actor.
        A single rollout is performed across each parallel environment
        Returns:
            A dictionary containing aggregated evaluation metrics.
        """
        print("\n--- Starting Evaluation ---")

        # Reset all parallel environments
        eval_env = self.vec_env.env
        eval_env.reset_idx(torch.arange(self.num_actors, device=self.device))
        # call compute observations to refresh physics tensors and then call env reset to get observations
        eval_env.compute_observations()
        obs = self.env_reset()

        
        if isinstance(obs, dict):
            obs = obs['obs']

        # reset logging from before evaluate was called
        eval_env.cumulatives['reward'][:] = 0
        eval_env.cumulatives['success'][:] = 0

        # Initialize tensors to store episode rewards and successes
        episode_rewards = torch.zeros(self.num_actors, device=self.device)
        success_count_per_episode = torch.zeros(self.num_actors, device=self.device)

        # --- Run one full rollout for all parallel environments ---
        for _ in range(eval_env.max_episode_length):
            with torch.no_grad(), autocast(
                device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
                if isinstance(obs, dict):
                    obs = obs['obs']
                # obs = self.obs_normalizer(obs,update=False)
                obs = self.preproc_obs(obs)
                actions = self.model.actor(obs) # no noise in evaluation

                next_obs, rewards, dones, infos = self.env_step(actions)

            if 'episode' in infos:
                success_count_per_episode = infos['episode']['success_count_per_episode']

            episode_rewards += rewards.squeeze(-1)
            obs = next_obs.copy()  
        
        # --- Aggregate and Report Final Metrics ---
        eval_metrics = {}
        print("--- Evaluation Summary ---")
        
        # Iterate through each task
        for task_idx in torch.unique(self.all_task_indices):
            task_name = self.ordered_task_names[task_idx.item()]
            
            # Find all parallel environments that correspond to this task
            task_env_indices = (self.all_task_indices == task_idx).nonzero(as_tuple=False).squeeze(-1)

            if task_env_indices.numel() > 0:
                # Calculate the mean success and reward for this task's trials
                task_success_rate = (success_count_per_episode[task_env_indices]>0).float().mean().item()
                task_avg_reward = episode_rewards[task_env_indices].mean().item()
                
                # Store metrics for logging
                eval_metrics[f'eval/{task_name}/success_rate'] = task_success_rate
                eval_metrics[f'eval/{task_name}/avg_reward'] = task_avg_reward
                
                print(f"Task: {task_name:<25} | Avg Success Rate: {task_success_rate:.4f} | Avg Reward: {task_avg_reward:.4f}")
        
        # Calculate overall average success rate across all tasks and trials
        overall_avg_success = (success_count_per_episode>0).float().mean().item()
        eval_metrics['eval/overall_success_rate'] = overall_avg_success
        eval_metrics['eval/overall_avg_reward'] = episode_rewards.mean().item()

        print(f"\nOverall Average Success Rate: {overall_avg_success:.4f}\n")
        print(f"Overall Average Reward: {episode_rewards.mean().item():.4f}\n")

        print("\n--- Evaluation Complete ---\n")

        return eval_metrics

    def play_steps(self, horizon, random_exploration):
        total_time_start = time.time()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        actor_losses = []
        critic1_loss_list = []
        critic2_loss_list = []
        critic_grad_norm_list = []
        actor_grad_norm_list = []
        train_metrics = {}
        eval_metrics = {}

        obs_dim = self.env_info["observation_space"].shape[0]
        action_dim = self.env_info["action_space"].shape[0]
        traj_obs = torch.empty((self.num_actors, horizon) + (obs_dim,), device=self.device)
        traj_actions = torch.empty((self.num_actors, horizon) + (action_dim,), device=self.device)
        traj_rewards = torch.empty((self.num_actors, horizon), device=self.device)
        traj_next_obs = torch.empty((self.num_actors, horizon) + (obs_dim,), device=self.device)
        traj_dones = torch.empty((self.num_actors, horizon), device=self.device)

        success_count_per_episode = torch.zeros(self.num_actors, device=self.device)

        obs = self.obs
        if isinstance(obs, dict):
            obs = self.obs['obs']

        next_obs_processed = obs.clone()

        for s in range(horizon):
            with torch.no_grad(),autocast(
                    device_type=self.amp_device_type, dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
                if random_exploration:
                    print(f"Warmup Step: {s}")
                    action = torch.rand((self.num_actors, *self.env_info["action_space"].shape), device=self._device) * 2.0 - 1.0
                else:
                    action = self.act(obs.float())

                step_start = time.time()
                next_obs, rewards, dones, infos = self.env_step(action)
                if 'episode' in infos:
                    success_count_per_episode = infos['episode']['success_count_per_episode']

                self.dones = dones
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

            self.obs = next_obs.copy() # changed from clone

            if isinstance(obs, dict):
                obs = self.obs['obs']

            traj_obs[:, s] = obs
            traj_actions[:, s] = action
            traj_dones[:, s] = dones
            traj_rewards[:, s] = rewards
            traj_next_obs[:, s] = next_obs_processed

        traj_rewards = traj_rewards.reshape(self.num_actors,horizon, 1)
        traj_dones = traj_dones.reshape(self.num_actors, horizon, 1)
        assert self.nstep >= horizon, "nstep needs to be greater than or equal to horizon"
        data = self.n_step_buffer.add_to_buffer(traj_obs, traj_actions, traj_rewards, traj_next_obs, traj_dones)

        self.replay_buffer.add_to_buffer(data)

        if not random_exploration:
            self.set_train()
            update_time_start = time.time()
            for gradient_step in range(self.gradient_steps_per_itr):
                actor_loss_info, critic1_loss, critic2_loss, critic_grad_norm, actor_grad_norm = self.update(self.epoch_num, gradient_step)
                if gradient_step % self.actor_update_freq == 1:
                    actor_grad_norm_list.append(actor_grad_norm)
                    self.extract_actor_stats(actor_losses, actor_loss_info)
            
            self.num_updates += self.gradient_steps_per_itr
            update_time_end = time.time()
            update_time = update_time_end - update_time_start

            critic1_loss_list.append(critic1_loss)
            critic2_loss_list.append(critic2_loss)
            critic_grad_norm_list.append(critic_grad_norm)
            
        else:
            update_time = 0

        total_update_time += update_time

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        # record training metrics before evaluation because evaluation will reset the environment
        if 'episode' in infos: 
            for task_idx in torch.unique(self.all_task_indices):
                task_name = self.ordered_task_names[task_idx.item()]
                
                task_env_indices = (self.all_task_indices == task_idx).nonzero(as_tuple=False).squeeze(-1)

                if task_env_indices.numel() > 0:
                    task_success_rate = (success_count_per_episode[task_env_indices]>0).float().mean().item()
                    # task_avg_reward = episode_rewards[task_env_indices].mean().item()
                    
                    train_metrics[f'train/{task_name}/success_rate'] = task_success_rate
                    # train_metrics[f'train/{task_name}/avg_reward'] = task_avg_reward
                    
                    # print(f"Task: {task_name:<25} | Avg Success Rate: {task_success_rate:.4f}")
            overall_avg_success = (success_count_per_episode>0).float().mean().item()
            train_metrics['train/overall_success_rate'] = overall_avg_success
            # train_metrics['train/overall_avg_reward'] = episode_rewards.mean().item()

            # print(f"\nOverall Average Success Rate: {overall_avg_success:.4f}\n")
            # print(f"Overall Average Reward: {episode_rewards.mean().item():.4f}\n")

        if self.epoch_num % self.evaluation_interval == 0: 
            self.set_eval()
            eval_metrics = self.evaluate()
            # hacky way to reset the environment after evaluation
            self.vec_env.env.reset_idx(torch.arange(self.num_actors, device=self.device))
            self.vec_env.env.compute_observations()
            self.obs = self.env_reset()

            # randomize progress_buf again
            # total_count = 0
            # for tid, env_count in zip(self.vec_env.env.task_idx, self.vec_env.env.task_env_count):
            #     self.vec_env.env.env_id_to_task_id[total_count:total_count+env_count] = [tid for _ in range(env_count)]
            #     if not self.vec_env.env.init_at_random_progress or tid in self.vec_env.env.exempted_init_at_random_progress_tasks:
            #         self.vec_env.env.progress_buf[total_count:total_count+env_count] = 0
            #     else:
            #         self.vec_env.env.progress_buf[total_count:total_count+env_count] = torch.randint_like(self.vec_env.env.progress_buf[total_count:total_count+env_count], high=int(self.vec_env.env.max_episode_length))
            
            #     total_count += env_count
            
            # self.game_rewards.clear()
            # self.game_lengths.clear()
            # self.mean_rewards = self.last_mean_rewards = -1000000000
        self.set_train()

        return step_time, play_time, total_update_time, total_time, actor_losses, critic1_loss_list, critic2_loss_list, critic_grad_norm_list, actor_grad_norm_list, train_metrics, eval_metrics

    def train_epoch(self):
        if self.epoch_num == 1:
            return self.play_steps(self.num_warmup_steps, True)
        else:
            return self.play_steps(self.horizon, False) 

    def train(self):
        self.init_tensors()
        self.algo_observer.after_init(self)
        total_time = 0

        self.obs = self.env_reset()

        while True:
            self.epoch_num += 1
            step_time, play_time, update_time, epoch_total_time, actor_losses, critic1_losses, critic2_losses, critic_grad_norms, actor_grad_norms, train_metrics,eval_metrics = self.train_epoch()

            self.actor_scheduler.step()
            self.critic_scheduler.step()

            total_time += epoch_total_time

            curr_frames = self.num_frames_per_epoch
            self.frame += curr_frames

            fps_step = curr_frames / step_time
            fps_step_inference = curr_frames / play_time
            fps_total = curr_frames / epoch_total_time

            print_statistics(self.print_stats, curr_frames, step_time, play_time, epoch_total_time, 
                self.epoch_num, self.max_epochs, self.frame, self.max_frames)

            self.writer.add_scalar('performance/step_inference_rl_update_fps', fps_total, self.frame)
            self.writer.add_scalar('performance/step_inference_fps', fps_step_inference, self.frame)
            self.writer.add_scalar('performance/step_fps', fps_step, self.frame)
            self.writer.add_scalar('performance/rl_update_time', update_time, self.frame)
            self.writer.add_scalar('performance/step_inference_time', play_time, self.frame)
            self.writer.add_scalar('performance/step_time', step_time, self.frame)

            # after the first epoch, we can start logging metrics
            if self.epoch_num > 1 :
                self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(actor_losses).item(), self.frame)
                self.writer.add_scalar('losses/c1_loss', torch_ext.mean_list(critic1_losses).item(), self.frame)
                self.writer.add_scalar('losses/c2_loss', torch_ext.mean_list(critic2_losses).item(), self.frame)

                for key,value in train_metrics.items():
                    self.writer.add_scalar(key, value, self.frame)
                
                # add eval metrics
                for key, value in eval_metrics.items():
                    self.writer.add_scalar(key, value, self.frame)
                # wandb.log(
                #     {   
                #         "performance/step_inference_rl_update_fps": fps_total,
                #         "performance/step_inference_fps": fps_step_inference,
                #         "performance/step_fps": fps_step,
                #         "performance/rl_update_time": update_time,
                #         "performance/step_inference_time": play_time,
                #         "performance/step_time": step_time,

                #         "losses/a_loss": torch_ext.mean_list(actor_losses).item(),
                #         "losses/c1_loss": torch_ext.mean_list(critic1_losses).item(),
                #         "losses/c2_loss": torch_ext.mean_list(critic2_losses).item(),
                #         "losses/critic_grad_norm": torch_ext.mean_list(critic_grad_norms).item(),
                #         "losses/actor_grad_norm": torch_ext.mean_list(actor_grad_norms).item(),
                #         "frame": self.frame,
                #     },
                #     step=self.frame,
                # )
                # if train_metrics:
                #     wandb.log(
                #         {   
                #             **train_metrics,
                #         },
                #         step=self.frame,
                #     )
            
                

            self.writer.add_scalar('info/epochs', self.epoch_num, self.frame)
            self.algo_observer.after_print_stats(self.frame, self.epoch_num, total_time)

            if self.game_rewards.current_size > 0:
                mean_rewards = self.game_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()

                self.writer.add_scalar('rewards/step', mean_rewards, self.frame)
                self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                self.writer.add_scalar('episode_lengths/step', mean_lengths, self.frame)
                self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)
                checkpoint_name = self.config['name'] + '_ep_' + str(self.epoch_num) + '_rew_' + str(mean_rewards)

                should_exit = False

                if self.save_freq > 0:
                    if self.epoch_num % self.save_freq == 0:
                        self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                if mean_rewards > self.last_mean_rewards and self.epoch_num >= self.save_best_after:
                    print('saving next best rewards: ', mean_rewards)
                    self.last_mean_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, self.config['name']))
                    if self.last_mean_rewards > self.config.get('score_to_win', float('inf')):
                        print('Maximum reward achieved. Network won!')
                        self.save(os.path.join(self.nn_dir, checkpoint_name))
                        should_exit = True

                if self.epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(self.epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

                if should_exit:
                    return self.last_mean_rewards, self.epoch_num


class MTFastTD3Agent(FastTD3Agent):
    def __init__(self, base_name, params):
        self.config = config = params['config']
        print(config)

        self.load_networks(params)
        self.base_init(base_name, config)
        self.num_warmup_steps = config["num_warmup_steps"]
        self.gamma = config["gamma"]
        self.critic_tau = float(config["critic_tau"])
        self.batch_size = config["batch_size"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.horizon = config["horizon"]
        self.normalize_input = config.get("normalize_input", False)
        self.normalize_reward = config.get('normalize_reward', False)
        self.gradient_steps_per_itr = config["gradient_steps_per_itr"]
        self.use_grad_norm = config['use_grad_norm']
        self.grad_norm = config.get('grad_norm', 0.5)
        self.noise_clip = config["noise_clip"]
        self.policy_noise = config["policy_noise"]
        self.actor_update_freq = config["actor_update_freq"]
        self.use_cdq = config["use_cdq"]
        self.nstep = config.get("nstep", 1)
        self.disable_bootstrap = config.get("disable_bootstrap", False)
        amp = config.get("amp", False)
        amp_dtype = config.get("amp_dtype", "bf16")
        self.amp_enabled = amp and torch.cuda.is_available()
        self.amp_device_type = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

        self.scaler = GradScaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)

        self.use_replay_ratio_scaling = config.get("use_replay_ratio_scaling", False)
        self.replay_ratio_scaling_update_freq = config.get("replay_ratio_scaling_update_freq", None)
        self.evaluation_interval = config.get("evaluation_interval", 1000)

        # TODO: double-check! To use bootstrap instead?
        self.max_env_steps = config.get("max_env_steps", 1000) # temporary, in future we will use other approach

        print(self.batch_size, self.num_actors, self.num_agents)

        self.num_frames_per_epoch = self.num_actors * self.horizon
        self.num_updates = 0

        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)

        self.all_task_indices : torch.Tensor = self.vec_env.env.extras["task_indices"]
        self.ordered_task_names : list[str] = self.vec_env.env.extras["ordered_task_names"]

        learn_task_embedding = config.get('learn_task_embedding', False)
        if learn_task_embedding:
            self.task_embedding_dim = config['task_embedding_dim']
        else: 
            self.task_embedding_dim = torch.unique(self.all_task_indices).shape[0]
        self.num_tasks = torch.unique(self.all_task_indices).shape[0]

        net_config = {
            'action_shape' : self.env_info["action_space"].shape,
            'input_shape' : obs_shape,
            'num_envs': self.num_actors,
            'device': self.device,
            'normalize_input': self.normalize_input,
            'learn_task_embedding': learn_task_embedding,
            'task_indices': self.all_task_indices,
            'task_embedding_dim': self.task_embedding_dim,
        }
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)

        self.actor_optimizer = torch.optim.AdamW(self.model.td3_network.actor.parameters(),
                                                lr=float(self.config['actor_lr']),
                                                betas=self.config.get("actor_betas", [0.9, 0.999]),
                                                weight_decay=self.config.get("weight_decay", 0.0))

        self.critic_optimizer = torch.optim.AdamW(self.model.td3_network.critic.parameters(),
                                                 lr=float(self.config["critic_lr"]),
                                                 betas=self.config.get("critic_betas", [0.9, 0.999]),
                                                 weight_decay=self.config.get("weight_decay", 0.0))

        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer,
            T_max=self.max_epochs,
            eta_min=float(self.config['critic_learning_rate_end'])
        )
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            T_max=self.max_epochs,
            eta_min=float(self.config['actor_learning_rate_end'])
        )

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

        # if self.normalize_input:
        #     self.obs_normalizer = EmpiricalNormalization(obs_shape[0], device=self.ppo_device).to(self.ppo_device)
        # else:
        #     self.obs_normalizer = torch.nn.Identity()

        if self.normalize_reward:
            self.reward_normalizer = PerTaskRewardNormalizer(torch.unique(self.all_task_indices).shape[0], self.gamma, device=self.ppo_device).to(self.ppo_device)

        # wandb.init(
        #     project="IsaacGym",
        #     name=config['name'] + datetime.now().strftime("_%d-%H-%M-%S"),
        #     save_code=True
        # )
    
    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs['obs']
        task_ids_one_hot = obs[...,-self.num_tasks:]
        task_indices = torch.argmax(task_ids_one_hot, dim=1)
        obs = self.model.norm_obs(obs, task_indices)
        return obs