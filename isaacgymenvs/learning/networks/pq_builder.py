import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from rl_games.algos_torch import network_builder
from isaacgymenvs.learning.networks.pq_network import QNetwork

from typing import List, Tuple

class PQNBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = PQNBuilder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            obs_dim = kwargs.pop('obs_dim')
            num_bins_per_dim = kwargs.pop('num_bins_per_dim')
            action_space_type = kwargs.pop('action_space_type')
            
            if 'task_indices' in kwargs:
                unique_task_indices = torch.unique(kwargs.pop('task_indices'))
                # task_embedding_dim = kwargs.pop('task_embedding_dim')

                # # get the dim of the real part of the obs
                # obs_dim = obs_dim - task_embedding_dim  
            else:
                unique_task_indices = None
                # task_embedding_dim = 0

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            critic_args = {
                'is_residual_network': self.is_residual_network,
                'input_size' : obs_dim, 
                'D' : self.D, 
                'num_blocks' : self.num_blocks,
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'norm_first_layer' : self.norm_first_layer,
                'unique_task_indices' : unique_task_indices,
                'action_space_type' : action_space_type
            }

            print("Building Critic")
            if action_space_type == "multi_discrete":
                # the output_dim is all state-action utilities 
                # for action dimensions n_a (action_dim) and discrete bins n_b (num_bins_per_dim)
                output_dim = num_bins_per_dim * actions_num
            else:
                output_dim = actions_num
            
            self.critic = self._build_critic(output_dim, num_bins_per_dim, None, critic_args)

        # the critic consists of a Q network
        def _build_critic(self, output_dim, num_bins_per_dim, task_encoder_args, critic_args):
            is_residual_network = critic_args['is_residual_network']
            in_features = critic_args['input_size']
            num_blocks = critic_args['num_blocks']
            hidden_features = critic_args['D']
            unique_task_indices = critic_args['unique_task_indices']
            action_space_type = critic_args['action_space_type']
            norm_first_layer = critic_args['norm_first_layer']

            Q = QNetwork(task_encoder_args, is_residual_network, action_space_type, num_bins_per_dim, in_features, output_dim, num_blocks, hidden_features, norm_first_layer, unique_task_indices)

            return Q

        def forward(self, obs_dict):
            """TODO"""
            pass
 
        def is_separate_critic(self):
            pass

        def load(self, params):
            self.is_residual_network = params['q']['residual_network']
            self.D = params['q']['D']
            self.num_blocks = params['q']['num_blocks']
            self.activation = params['q']['activation']
            self.initializer = params['q']['initializer']
            self.is_d2rl = params['q'].get('d2rl', False)
            self.norm_first_layer = params['q'].get('norm_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params

            if self.has_space:
                self.is_discrete = 'discrete' in params['space']
                self.multi_discrete = 'multi_discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.multi_discrete:
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.multi_discrete = False
    

