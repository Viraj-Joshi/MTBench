import torch
from torch import nn
import numpy as np

from collections import defaultdict

from rl_games.algos_torch.models import ModelA2CContinuousLogStd, ModelSACContinuous, BaseModel
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.algos_torch.sac_helper import SquashedNormal
import rl_games.common.divergence as divergence
from rl_games.algos_torch.models import BaseModelNetwork

""" class MTRunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        self.kwargs = {'insize': insize, 'epsilon': epsilon, 'per_channel': per_channel, 'norm_only': norm_only}

    def forward(self, input, task_id, denorm=False, mask=None): """

class MTModelNetwork(nn.Module):
    def __init__(self, obs_shape, normalize_value, normalize_input, value_size, task_embedding_dim):
        nn.Module.__init__(self)
        
        obs_dim = obs_shape[0]  # true obs dim + task embedding dim
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input
        self.value_size = value_size
        self.task_embedding_dim = task_embedding_dim # is also equivalent to the number of tasks when using a one-hot encoding

        self.true_obs_dim = obs_dim - self.task_embedding_dim
        if normalize_value:
            self.value_mean_stds = nn.ModuleList([RunningMeanStd((self.value_size,)) for _ in range(task_embedding_dim)]) #   GeneralizedMovingStats((self.value_size,)) #   
        if normalize_input:
            if isinstance(obs_shape, dict):
                self.running_mean_stds = nn.ModuleList([RunningMeanStdObs(self.true_obs_dim) for _ in range(task_embedding_dim)])
            else:
                self.running_mean_stds = nn.ModuleList([RunningMeanStd(self.true_obs_dim) for _ in range(task_embedding_dim)])
        
    def norm_obs(self, observation, task_indices):
        if observation.shape[0] != task_indices.shape[0]:
            raise ValueError(f"number of observations {observation.shape[0]} does not match number of task indices {task_indices.shape[0]}")
        if self.normalize_input:
            true_obs = observation[:, :self.true_obs_dim]
            task_indices = task_indices.squeeze(-1)
            with torch.no_grad():
                current_mean = torch.zeros_like(true_obs)
                current_var = torch.ones_like(true_obs)
                for tid in torch.unique(task_indices):
                    idx = (task_indices == tid).nonzero(as_tuple=False).squeeze(-1)
                    if self.running_mean_stds.training:
                        self.running_mean_stds[tid.item()](true_obs[idx])
                    current_mean[idx] = self.running_mean_stds[tid.item()].running_mean.float()
                    current_var[idx] = self.running_mean_stds[tid.item()].running_var.float()
                normalized_obs  = (true_obs - current_mean) / (current_var + 1e-8).sqrt()
                return torch.cat([normalized_obs, observation[:, self.true_obs_dim:]], dim=1)
        else:
            return observation

    def denorm_value(self, value, task_indices):
        with torch.no_grad():
            # return self.value_mean_std(value, denorm=True) if self.normalize_value else value
            if self.normalize_value:
                for tid in torch.unique(task_indices):
                    mask = task_indices == tid
                    value[mask] = self.value_mean_stds[tid.item()](value[mask], denorm=True)
                return value
            else:
                return value

# taken from FastTD3: https://github.com/younggyoseo/FastTD3
# need per-task normalization for the rewards for distributional critic
class PerTaskEmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values for each task."""

    def __init__(
        self,
        num_tasks: int,
        shape: tuple,
        device: torch.device,
        eps: float = 1e-2,
        until: int = None,
    ):
        """
        Initialize PerTaskEmpiricalNormalization module.

        Args:
            num_tasks (int): The total number of tasks.
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If specified, learns until the sum of batch sizes
                                 for a specific task exceeds this value.
        """
        super().__init__()
        if not isinstance(shape, tuple):
            shape = (shape,)
        self.num_tasks = num_tasks
        self.shape = shape
        self.eps = eps
        self.until = until
        self.device = device

        # Buffers now have a leading dimension for tasks
        self.register_buffer("_mean", torch.zeros(num_tasks, *shape).to(device))
        self.register_buffer("_var", torch.ones(num_tasks, *shape).to(device))
        self.register_buffer("_std", torch.ones(num_tasks, *shape).to(device))
        self.register_buffer(
            "count", torch.zeros(num_tasks, dtype=torch.long).to(device)
        )

    def forward(
        self, x: torch.Tensor, task_ids: torch.Tensor, center: bool = True
    ) -> torch.Tensor:
        """
        Normalize the input tensor `x` using statistics for the given `task_ids`.

        Args:
            x (torch.Tensor): Input tensor of shape [num_envs, *shape].
            task_ids (torch.Tensor): Tensor of task indices, shape [num_envs].
            center (bool): If True, center the data by subtracting the mean.
        """
        if x.shape[1:] != self.shape:
            raise ValueError(f"Expected input shape (*, {self.shape}), got {x.shape}")
        if x.shape[0] != task_ids.shape[0]:
            raise ValueError("Batch size of x and task_ids must match.")

        # Gather the stats for the tasks in the current batch
        # Reshape task_ids for broadcasting: [num_envs] -> [num_envs, 1, ...]
        view_shape = (task_ids.shape[0],) + (1,) * len(self.shape)
        task_ids_expanded = task_ids.view(view_shape).expand_as(x)

        mean = self._mean.gather(0, task_ids_expanded)
        std = self._std.gather(0, task_ids_expanded)

        if self.training:
            self.update(x, task_ids)

        if center:
            return (x - mean) / (std + self.eps)
        else:
            return x / (std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor, task_ids: torch.Tensor):
        """Update running statistics for the tasks present in the batch."""
        unique_tasks = torch.unique(task_ids)

        for task_id in unique_tasks:
            if self.until is not None and self.count[task_id] >= self.until:
                continue

            # Create a mask to select data for the current task
            mask = task_ids == task_id
            x_task = x[mask]
            batch_size = x_task.shape[0]

            if batch_size == 0:
                continue

            # Update count for this task
            old_count = self.count[task_id].clone()
            new_count = old_count + batch_size

            # Update mean
            task_mean = self._mean[task_id]
            batch_mean = torch.mean(x_task, dim=0)
            delta = batch_mean - task_mean
            self._mean[task_id].copy_(task_mean + (batch_size / new_count) * delta)

            # Update variance using Chan's parallel algorithm
            if old_count > 0:
                batch_var = torch.var(x_task, dim=0, unbiased=False)
                m_a = self._var[task_id] * old_count
                m_b = batch_var * batch_size
                M2 = m_a + m_b + (delta**2) * (old_count * batch_size / new_count)
                self._var[task_id].copy_(M2 / new_count)
            else:
                # For the first batch of this task
                self._var[task_id].copy_(torch.var(x_task, dim=0, unbiased=False))

            self._std[task_id].copy_(torch.sqrt(self._var[task_id]))
            self.count[task_id].copy_(new_count)


class PerTaskRewardNormalizer(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        gamma: float,
        device: torch.device,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        """
        Per-task reward normalizer, motivation comes from BRC (https://arxiv.org/abs/2505.23150v1)
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon
        self.device = device

        # Per-task running estimate of the discounted return
        self.register_buffer("G", torch.zeros(num_tasks, device=device))
        # Per-task running-max of the discounted return
        self.register_buffer("G_r_max", torch.zeros(num_tasks, device=device))
        # Use the new per-task normalizer for the statistics of G
        self.G_rms = PerTaskEmpiricalNormalization(
            num_tasks=num_tasks, shape=(1,), device=device
        )

    def _scale_reward(
        self, rewards: torch.Tensor, task_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Scales rewards using per-task statistics.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        # Gather stats for the tasks in the batch
        std_for_batch = self.G_rms._std.gather(0, task_ids.unsqueeze(-1)).squeeze(-1)
        g_r_max_for_batch = self.G_r_max.gather(0, task_ids)

        var_denominator = std_for_batch + self.epsilon
        min_required_denominator = g_r_max_for_batch / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)

        # Add a small epsilon to the final denominator to prevent division by zero
        # in case g_r_max is also zero.
        return rewards / (denominator + self.epsilon)

    def update_stats(
        self, rewards: torch.Tensor, dones: torch.Tensor, task_ids: torch.Tensor
    ):
        """
        Updates the running discounted return and its statistics for each task.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            dones (torch.Tensor): Done tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        if not (rewards.shape == dones.shape == task_ids.shape):
            raise ValueError("rewards, dones, and task_ids must have the same shape.")

        # === Update G (running discounted return) ===
        # Gather the previous G values for the tasks in the batch
        prev_G = self.G.gather(0, task_ids)
        # Update G for each environment based on its own reward and done signal
        new_G = self.gamma * (1 - dones.float()) * prev_G + rewards
        # Scatter the updated G values back to the main buffer
        self.G.scatter_(0, task_ids, new_G)

        # === Update G_rms (statistics of G) ===
        # The update function handles the per-task logic internally
        self.G_rms.update(new_G.unsqueeze(-1), task_ids)

        # === Update G_r_max (running max of |G|) ===
        prev_G_r_max = self.G_r_max.gather(0, task_ids)
        # Update the max for each environment
        updated_G_r_max = torch.maximum(prev_G_r_max, torch.abs(new_G))
        # Scatter the new maxes back to the main buffer
        self.G_r_max.scatter_(0, task_ids, updated_G_r_max)

    def forward(self, rewards: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Normalizes rewards. During training, it also updates the running statistics.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        return self._scale_reward(rewards, task_ids)    

class MTModelA2CContinuousLogStd(ModelA2CContinuousLogStd):
    # Multitask variance of the continuous_a2c_logstd model

    def __init__(self, network):
        super().__init__(network)

    def build(self, config):
        obs_shape = config['input_shape']
        normalize_value = config['normalize_value']
        normalize_input = config['normalize_input']
        value_size = config.get('value_size', 1)
        if 'task_indices' not in config:
            raise KeyError("task_indices not found for a multi task model")
        task_indices = config["task_indices"]
        task_embedding_dim = torch.unique(task_indices).shape[0]
        return self.Network(self.network_builder.build(self.model_class, **config), obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size, task_embedding_dim=task_embedding_dim)

    class Network(MTModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            MTModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return self.a2c_network.is_rnn()
        
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()            

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            # added task_indices
            task_indices = input_dict.get('task_indices', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'], task_indices)
            # ---------------------
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)

            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,  # this is unnormalized value
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value, task_indices),
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result
            
        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)

import copy
class MTModelSACContinuous(ModelSACContinuous):

    def __init__(self, network):
        ModelSACContinuous.__init__(self, 'sac')
        self.network_builder = network
        
    def build(self, config):
        obs_shape = config['input_shape']
        normalize_value = config['normalize_value']
        normalize_input = config['normalize_input']
        value_size = config.get('value_size', 1)
        if 'task_indices' not in config:
            raise KeyError("task_indices not found for a multi task model")
        task_indices = config["task_indices"]
        task_embedding_dim = torch.unique(task_indices).shape[0]
        return self.Network(self.network_builder.build(self.model_class, **config), obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size, task_embedding_dim=task_embedding_dim)
    
    class Network(MTModelNetwork):
        def __init__(self, sac_network,**kwargs):
            MTModelNetwork.__init__(self,**kwargs)
            self.sac_network = sac_network
            self.original_sac_network = copy.deepcopy(sac_network)
            self.kwargs = kwargs

        def critic(self, obs, action):
            return self.sac_network.critic(obs, action)

        def critic_target(self, obs, action):
            return self.sac_network.critic_target(obs, action)

        def actor(self, obs):
            return self.sac_network.actor(obs)
        
        def is_rnn(self):
            return False

        # for scaling replay ratio
        def reset_all_parameters(self):
            self.__init__(self.original_sac_network, **self.kwargs)
            self.running_mean_stds = self.running_mean_stds.to('cuda')
            self.value_mean_stds = self.value_mean_stds.to('cuda')

        def forward(self, input_dict):
            is_train = input_dict.pop('is_train', True)
            mu, sigma = self.sac_network(input_dict)
            dist = SquashedNormal(mu, sigma)
            return dist
    
class ModelFastTD3(BaseModel):

    def __init__(self, network):
        BaseModel.__init__(self, 'td3')
        self.network_builder = network
    
    class Network(BaseModelNetwork):
        def __init__(self, td3_network,**kwargs):
            BaseModelNetwork.__init__(self,**kwargs)
            self.td3_network = td3_network
            self.kwargs = kwargs

        def norm_obs(self, observation):
            with torch.no_grad():
                return self.running_mean_std(observation) if self.normalize_input else observation
            
        def critic(self, obs, action):
            return self.td3_network.critic(obs, action)

        def critic_target(self, obs, action):
            return self.td3_network.critic_target(obs, action)

        def actor(self, obs):
            return self.td3_network.actor(obs)
        
        def is_rnn(self):
            return False

        def forward(self, input_dict):
            return self.td3_network(input_dict)

class MTModelFastTD3Continuous(BaseModel):

    def __init__(self, network):
        ModelFastTD3.__init__(self, network)
    
    def build(self, config):
        obs_shape = config['input_shape']
        normalize_input = config['normalize_input']
        if 'task_indices' not in config:
            raise KeyError("task_indices not found for a multi task model")
        task_indices = config["task_indices"]
        task_embedding_dim = torch.unique(task_indices).shape[0]
        return self.Network(self.network_builder.build(self.model_class, **config), obs_shape=obs_shape,
            normalize_value=False, normalize_input=normalize_input, value_size=1, task_embedding_dim=task_embedding_dim)

    class Network(MTModelNetwork):
        def __init__(self, td3_network,**kwargs):
            MTModelNetwork.__init__(self,**kwargs)
            self.td3_network = td3_network
            self.kwargs = kwargs

        def critic(self, obs, action):
            return self.td3_network.critic(obs, action)

        def critic_target(self, obs, action):
            return self.td3_network.critic_target(obs, action)

        def actor(self, obs):
            return self.td3_network.actor(obs)
        
        def is_rnn(self):
            return False

        def forward(self, input_dict):
            return self.td3_network(input_dict)
    

class ModelParallelQ(BaseModel):

    def __init__(self, network):
        BaseModel.__init__(self, 'parallel-q')
        self.network_builder = network
    
    class Network(BaseModelNetwork):
        def __init__(self, parallel_q_network, **kwargs):
            BaseModelNetwork.__init__(self,**kwargs)
            self.parallel_q_network = parallel_q_network

        def norm_obs(self, observation):
            with torch.no_grad():
                return self.running_mean_std(observation) if self.normalize_input else observation

        def forward(self, input_dict):
            return self.parallel_q_network(input_dict)

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def critic(self, obs):
            return self.parallel_q_network.critic(obs)

class MTModelParallelQ(BaseModel):

    def __init__(self, network):
        ModelParallelQ.__init__(self, network)
    
    def build(self, config):
        obs_shape = config['input_shape']
        normalize_value = False
        normalize_input = config['normalize_input']
        value_size = config.get('value_size', 1)
        if 'task_indices' not in config:
            raise KeyError("task embeddings must be true for a multi task model. check the task config file")
        task_indices = config["task_indices"]
        task_embedding_dim = torch.unique(task_indices).shape[0]
        return self.Network(self.network_builder.build(self.model_class, **config), obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size, task_embedding_dim=task_embedding_dim)

    class Network(MTModelNetwork):
        def __init__(self, parallel_q_network,**kwargs):
            MTModelNetwork.__init__(self,**kwargs)
            self.parallel_q_network = parallel_q_network

        def forward(self, input_dict):
            return self.parallel_q_network(input_dict)

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def critic(self, obs):
            return self.parallel_q_network.critic(obs)
        
class MTModelGRPOContinuousLogStd(BaseModel):
    def __init__(self, network,**kwargs):
        BaseModel.__init__(self, 'grpo')
        self.network_builder = network

    def build(self, config):
        obs_shape = config['input_shape']
        normalize_value = False                      # GRPO does not have a critic
        normalize_input = config['normalize_input']
        value_size = config.get('value_size', 1)
        if 'task_indices' not in config:
            raise KeyError("task_indices not found for a multi task model")
        task_indices = config["task_indices"]
        task_embedding_dim = torch.unique(task_indices).shape[0]
        return self.Network(self.network_builder.build(self.model_class, **config), obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size, task_embedding_dim=task_embedding_dim)

    class Network(MTModelNetwork):
        def __init__(self, grpo_network, **kwargs):
            MTModelNetwork.__init__(self, **kwargs)
            self.grpo_network = grpo_network

        def is_rnn(self):
            return self.grpo_network.is_rnn()
        
        def get_default_rnn_state(self):
            return self.grpo_network.get_default_rnn_state()            


        def kl(self, p_dict, q_dict):
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            # added task_indices
            task_indices = input_dict.get('task_indices', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'], task_indices)
            # ---------------------
            mu, logstd, states = self.grpo_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)

            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result
            
        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)
