from rl_games.algos_torch import torch_ext

from rl_games.common import vecenv
from rl_games.common import schedulers
from rl_games.common import experience
from rl_games.common.a2c_common import print_statistics

from rl_games.interfaces.base_algorithm import  BaseAlgorithm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from rl_games.algos_torch import  model_builder
from torch import optim
import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
import os

class PQNAgent(BaseAlgorithm):

    def __init__(self, base_name, params):

        self.config = config = params['config']
        print(config)

        self.load_networks(params)
        self.base_init(base_name, config)
        self.gamma = config["gamma"]
        self.num_minibatches = config["num_minibatches"]
        self.horizon = config["horizon"]
        self.normalize_input = config.get("normalize_input", False)
        self.mini_epochs = config.get("mini_epochs", 1)


        self.epsilon_decay = config.get("epsilon_decay", True)
        self.start_e: float = config['start_e']
        self.end_e: float = config['end_e']
        self.exploration_fraction: float = config['exploration_fraction']
        self.q_lambda: float = config["q_lambda"]
        self.max_grad_norm: float = config.get("max_grad_norm", 10.0)
        self.anneal_lr: bool = config.get("anneal_lr", False)

        # TODO: double-check! To use bootstrap instead?
        self.max_env_steps = config.get("max_env_steps", 1000) # temporary, in future we will use other approach

        print(self.num_actors, self.num_agents)

        self.num_frames_per_epoch = self.num_actors * self.horizon

        action_space = self.env_info['action_space']
        if hasattr(action_space, 'n'):
            self.actions_num = action_space.n                           # number of discrete choices for the 1d action
            self.action_dim = 1                                         # dim(action)
            self.action_space_type = "discrete"
        else:
            self.action_dim = self.actions_num = action_space.shape[0] # dim(action)
            self.num_bins_per_dim = action_space.nvec[0]               # actions per dim
            self.action_space_type = "multi_discrete"                  # pqn bins continuous actions per dim

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        
        net_config = {
            'obs_dim': obs_shape[0],
            'input_shape' : obs_shape,
            'num_bins_per_dim': self.num_bins_per_dim if self.action_space_type=="multi_discrete" else None,
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'action_space_type': self.action_space_type,
        }
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        print("Number of Agents", self.num_actors)

        self.critic_optimizer = torch.optim.RAdam(self.model.parallel_q_network.critic.parameters(),
                                                 lr=float(self.config["critic_lr"]),
                                                 betas=self.config.get("critic_betas", [0.9, 0.999]))

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

        self.rewards_shaper = config['reward_shaper']
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        #self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.c_loss = nn.MSELoss()

        self.save_best_after = config.get('save_best_after', 500)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.save_freq = config.get('save_frequency', 0)

        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
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

        self.dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self._device)

    @property
    def device(self):
        return self._device

    def get_weights(self):
        print("Loading weights")
        state = {'critic': self.model.parallel_q_network.critic.state_dict(), }
        return state

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def set_weights(self, weights):
        self.model.parallel_q_network.critic.load_state_dict(weights['critic'])

        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])

    def get_full_state_weights(self):
        print("Loading full weights")
        state = self.get_weights()

        state['epoch'] = self.epoch_num
        state['frame'] = self.frame
        state['critic_optimizer'] = self.critic_optimizer.state_dict()  

        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)

        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']

        self.critic_optimizer.load_state_dict(weights['critic_optimizer'])

        self.last_mean_rewards = weights.get('last_mean_rewards', -1000000000)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

    def restore(self, fn, set_epoch=True):
        print("PQN restore")
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
        obs = self.preproc_obs(obs)
        q_values = self.model.critic(obs)
        # q_values shape: (num_envs, action_dim) or (num_envs, action_dim, num_bins_per_dim)

        if self.action_space_type == "multi_discrete":
            max_q_values, max_actions = torch.max(q_values, dim=2) # get max state-action per action dimension
            # max_q_values shape: (num_envs, action_dim)
        else:
            max_q_values, max_actions = torch.max(q_values, dim=1)
            # max_q_values shape: (num_envs)

        return max_q_values, max_actions

    def clear_stats(self):
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.algo_observer.after_clear_stats()
    
    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def play_steps(self):
        total_time_start = time.time()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        critic_losses = []

        obs_dim = self.env_info["observation_space"].shape[0]
        action_dim = self.action_dim
        num_envs = self.num_actors
        
        # store transitions
        obses = torch.zeros((self.horizon, num_envs, obs_dim),device=self._device,dtype=torch.float32)
        actions = torch.zeros((self.horizon, num_envs, action_dim),device=self._device,dtype=torch.float32)
        values =  torch.zeros((self.horizon, num_envs),device=self._device,dtype=torch.float32)
        rewards = torch.zeros((self.horizon, num_envs),device=self._device,dtype=torch.float32)
        dones = torch.zeros((self.horizon, num_envs),device=self._device,dtype=torch.bool)

        obs = self.obs
        if isinstance(obs, dict):
            obs = self.obs['obs']

        next_obs_processed = obs.clone()

        done = torch.zeros((num_envs,), device=self._device, dtype=torch.bool)
        total_timesteps = self.max_epochs * self.num_actors * self.horizon

        if self.anneal_lr:
            frac = 1.0 - (self.epoch_num - 1.0) / self.max_epochs
            lrnow = frac * float(self.config["critic_lr"])
            self.critic_optimizer.param_groups[0]["lr"] = lrnow

        for s in range(self.horizon):
            self.set_eval()

            obses[s] = obs
            dones[s] = done

            if self.epsilon_decay:
                epsilon = self.linear_schedule(self.start_e, self.end_e, self.exploration_fraction * total_timesteps, (self.epoch_num-1)*self.num_actors + self.num_actors * (s+1))
            else:
                epsilon = self.end_e
            if self.action_space_type == "multi_discrete":
                random_action = torch.randint(0, self.num_bins_per_dim, 
                                            (num_envs, action_dim), 
                                            device=self._device, dtype=torch.float32)
            else:
                random_action = torch.randint(0, action_dim, 
                                            (num_envs,), 
                                            device=self._device, dtype=torch.float32)
            
            with torch.no_grad():
                max_q_values, max_action = self.act(obs.float())
                if self.action_space_type == "multi_discrete":
                    # above returns shapes (num_envs, action_dim), (num_envs, action_dim)
                    values[s] = max_q_values.mean(dim=-1) # average decoupled state-action values
                    # RHS is of shape (num_envs) 
                else:
                    values[s] = max_q_values

            explore = (torch.rand((num_envs,), device=self._device) < epsilon)
            if self.action_space_type == "multi_discrete":
                explore = explore.unsqueeze(-1).expand(-1, action_dim)
            
            action = torch.where(explore, random_action, max_action)
            
            if self.action_space_type == "discrete":
                action = action.unsqueeze(-1)
            actions[s] = action

            step_start = time.time()
            if self.action_space_type == "multi_discrete":
                if self.num_bins_per_dim == 2:
                    env_action = torch.where(action==0, -1.0, 1.0)
                elif self.num_bins_per_dim == 3:
                    env_action = action.clone()-1  # subtract 1 to create action in range [-1,1]
                else:
                    spacing = 2.0 / (self.num_bins_per_dim - 1)  # Total range (2) divided by (bins - 1)
                    env_action = action * spacing - 1.0
            else:
                if self.actions_num == 2:
                    env_action = torch.where(action==0, -1, action)
                elif self.actions_num == 3:
                    env_action = action.clone()-1
            
            with torch.no_grad(): # reward will have a gradient otherwise
                next_obs, reward, done, infos = self.env_step(env_action)
            step_end = time.time()

            rewards[s] = reward

            self.current_rewards += reward
            self.current_lengths += 1

            total_time += (step_end - step_start)
            step_time += (step_end - step_start)

            all_done_indices = done.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            not_done = 1.0 - done.float()

            self.algo_observer.process_infos(infos, done_indices)

            no_timeouts = self.current_lengths != self.max_env_steps
            done = done * no_timeouts

            self.current_rewards = self.current_rewards * not_done
            self.current_lengths = self.current_lengths * not_done

            if isinstance(next_obs, dict):    
                next_obs_processed = next_obs['obs']

            self.obs = next_obs.copy() # changed from .clone()

            rewards = self.rewards_shaper(rewards)

            if isinstance(obs, dict):
                obs = self.obs['obs']

        # Compute targets following Q(lambda)
        with torch.no_grad():
            returns = torch.zeros_like(rewards)
            for t in reversed(range(self.horizon)):
                if t == self.horizon - 1:
                    next_values, _ = self.act(next_obs)
                    if self.action_space_type == "multi_discrete":
                        next_values = next_values.mean(dim=-1) # average decoupled state-action values
                    nextnonterminal = not_done
                    returns[t] = rewards[t] + self.gamma * next_values * nextnonterminal
                else:
                    nextnonterminal = 1 - dones[t + 1].float()
                    next_values = values[t + 1]
                    returns[t] = rewards[t] + self.gamma * (
                        self.q_lambda * returns[t + 1] + (1 - self.q_lambda) * next_values * nextnonterminal
                    ) 

        # Optimize the Q-network
        self.set_train()
        batch_size = self.horizon * num_envs
        minibatch_size = batch_size // self.num_minibatches

        # flatten the batch
        b_obses = obses.reshape(-1, obs_dim)
        b_actions = actions.reshape(-1, action_dim)
        b_returns = returns.reshape(-1)

        b_inds = torch.arange(batch_size, device=b_obses.device)
        for _ in range(self.mini_epochs):
            update_time_start = time.time()
            b_inds = torch.randperm(batch_size)
            b_critic_losses = []
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                if self.action_space_type == "multi_discrete":
                    chosen_q = self.model.critic(b_obses[mb_inds]).gather(2, b_actions[mb_inds].long().unsqueeze(-1)).squeeze(-1) 
                    # shape: (minibatch_size, action_dim)
                    old_val = chosen_q.mean(dim=-1)  # shape: (minibatch_size,)
                else:
                    old_val = self.model.critic(b_obses[mb_inds]).gather(1, b_actions[mb_inds].long()).squeeze(-1)
                
                critic_loss = self.c_loss(b_returns[mb_inds], old_val)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parallel_q_network.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                b_critic_losses.append(critic_loss.detach())
            
            update_time_end = time.time()
            update_time = update_time_end - update_time_start

            critic_losses.append(torch.mean(torch.stack(b_critic_losses)))

            total_update_time += update_time

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        return step_time, play_time, total_update_time, total_time, critic_losses, old_val.mean().item(), epsilon

    def train_epoch(self):
        return self.play_steps()

    def train(self):
        self.init_tensors()
        self.algo_observer.after_init(self)
        total_time = 0

        self.obs = self.env_reset()

        while True:
            self.epoch_num += 1
            step_time, play_time, update_time, epoch_total_time, critic_losses, critic_values, epsilon = self.train_epoch()

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

            self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(critic_losses).item(), self.frame)
            self.writer.add_scalar('info/critic_value', critic_values, self.frame)

            self.writer.add_scalar('info/epochs', self.epoch_num, self.frame)
            self.writer.add_scalar('info/epsilon', epsilon, self.frame)
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


class MTPQNAgent(PQNAgent):
    def __init__(self, base_name, params):

        self.config = config = params['config']
        print(config)

        # TODO: Get obs shape and self.network
        self.load_networks(params)
        self.base_init(base_name, config)
        self.gamma = config["gamma"]
        self.num_minibatches = config["num_minibatches"]
        self.horizon = config["horizon"]
        self.normalize_input = config.get("normalize_input", False)
        self.mini_epochs = config.get("mini_epochs", 1)


        self.epsilon_decay = config.get("epsilon_decay", True)
        self.start_e: float = config['start_e']
        self.end_e: float = config['end_e']
        self.exploration_fraction: float = config['exploration_fraction']
        self.q_lambda: float = config["q_lambda"]
        self.max_grad_norm: float = config.get("max_grad_norm", 10.0)
        self.anneal_lr: bool = config.get("anneal_lr", False)

        self.max_env_steps = config.get("max_env_steps", 1000) # temporary, in future we will use other approach

        print(self.num_actors, self.num_agents)

        self.num_frames_per_epoch = self.num_actors * self.horizon

        action_space = self.env_info['action_space']
        if hasattr(action_space, 'n'):
            self.actions_num = action_space.n                           # number of discrete choices for the 1d action
            self.action_dim = 1                                         # dim(action)
            self.action_space_type = "discrete"
        else:
            self.action_dim = self.actions_num = action_space.shape[0] # dim(action)
            self.num_bins_per_dim = action_space.nvec[0]               # actions per dim
            self.action_space_type = "multi_discrete"                  # pqn bins continuous actions per dim

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        
        self.all_task_indices = self.vec_env.env.extras["task_indices"]
        self.task_embedding_dim = torch.unique(self.all_task_indices).shape[0]
        
        net_config = {
            'obs_dim': obs_shape[0],
            'normalize_value' : False, # dummy variable for MTModelNetwork
            'normalize_input': self.normalize_input,
            'input_shape' : obs_shape,
            'num_bins_per_dim': self.num_bins_per_dim if self.action_space_type=="multi_discrete" else None,
            'actions_num' : self.actions_num,
            'action_space_type': self.action_space_type,
            'task_indices': self.all_task_indices,
            'task_embedding_dim': self.task_embedding_dim,
        }
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        print("Number of Agents", self.num_actors)

        self.critic_optimizer = torch.optim.RAdam(self.model.parallel_q_network.critic.parameters(),
                                                 lr=float(self.config["critic_lr"]),
                                                 betas=self.config.get("critic_betas", [0.9, 0.999]))

        self.algo_observer = config['features']['observer']

    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs['obs']
        obs = self.model.norm_obs(obs, self.all_task_indices)
        return obs