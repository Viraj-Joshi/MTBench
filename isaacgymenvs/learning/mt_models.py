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
            import ipdb; ipdb.set_trace()
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
            self.td3_network = sac_network
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