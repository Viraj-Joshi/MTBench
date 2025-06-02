import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from rl_games.algos_torch import network_builder

from typing import List, Tuple
from isaacgymenvs.learning.networks.soft_modularized_network import SoftModularizedMLPWrapper, SharedPPOSoftModularizedMLP, weight_init

class SoftModularizedA2CBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            action_dim = kwargs.pop('actions_num')
            obs_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)

            unique_task_indices = torch.unique(kwargs.pop('task_indices'))
            task_embedding_dim = kwargs.pop('task_embedding_dim')

            # get the dim of the real part of the obs
            true_obs_dim = obs_shape[0] - task_embedding_dim

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
        
            actor_args = {
                'input_size' : true_obs_dim, 
                'num_experts' : self.num_experts,
                'D' : self.D, 
                'num_layers' : self.num_layers,
                'unique_task_indices' : unique_task_indices,
                'fixed_sigma' : self.fixed_sigma,
                'd2rl' : self.is_d2rl,
            }

            state_encoder_args = {
                'input_size' : true_obs_dim, 
                'units' : self.state_encoder_units, 
                'activation' : self.state_encoder_activation,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
            }

            task_encoder_args = {
                'input_size' : task_embedding_dim, # the in dimension of the MLP
                'units' : self.task_encoder_units, # hidden layer sizes
                'activation' : self.task_encoder_activation,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                # 'task_encoder_bias': False
            }

            head_args = {
                'units' : self.head_units,
                'activation' : self.head_activation,
                'initializer' : self.head_initializer,
                'output_dim': 1
            }
            
            if not self.separate:
                print("Building Shared Actor and Critic")
                actor_output_dim = action_dim
                critic_output_dim = self.value_size
                self.actor = SharedPPOSoftModularizedMLP(actor_output_dim, critic_output_dim, task_encoder_args, state_encoder_args, actor_args)

                self.actor.apply(weight_init)
            else:
                print("Building Separate Actor")
                self.actor = self._build_soft_modularized_network(action_dim, task_encoder_args, state_encoder_args, actor_args)
                self.actor.apply(weight_init)
                
                print("Building Separate Critic")    
                critic_args = actor_args.copy()
                self.critic = self._build_soft_modularized_network(self.value_size, task_encoder_args, state_encoder_args, critic_args)
                self.critic.apply(weight_init)
            
            if not self.is_continuous:
                raise ValueError("Only continuous actions are supported as of now")
            
            if self.is_continuous:
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(action_dim, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(self.D, action_dim)

            if self.is_continuous:
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)  

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('states', None)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)

            if self.separate:
                mu = self.actor(obs)
                value = self.critic(obs)

                if self.is_continuous:
                    if self.fixed_sigma:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        raise NotImplementedError("Only fixed sigma supported")
                        # sigma = self.sigma_act(self.sigma(a_out))

                    return mu, sigma, value, states
            else:      
                mu, value = self.actor(obs)

                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                
                return mu, mu*0 + sigma, value, states
        
        def _build_soft_modularized_network(self, output_dim, task_encoder_args, state_encoder_args, network_args):
            return SoftModularizedMLPWrapper(output_dim, task_encoder_args, state_encoder_args, network_args)
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            pass

        def get_default_rnn_state(self):
            pass
            
        def load(self, params):
            self.separate = params.get('separate', False)
            self.num_experts = params['soft_network']['num_experts']
            self.D = params['soft_network']['D']
            self.num_layers = params['soft_network']['num_layers']
            self.activation = params['soft_network']['activation']
            self.initializer = params['soft_network']['initializer']
            self.is_d2rl = params['soft_network'].get('d2rl', False)
            self.norm_only_first_layer = params['soft_network'].get('norm_only_first_layer', False)
            
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            self.head_units = params['head']['units']
            self.head_activation = params['head']['activation']
            self.head_initializer = params['head']['initializer']

            self.state_encoder_units = params['state_encoder']['units']
            self.state_encoder_activation = params['state_encoder']['activation']
            self.state_encoder_initializer = params['state_encoder']['initializer']

            self.task_encoder_units = params['task_encoder']['units']
            self.task_encoder_activation = params['task_encoder']['activation']
            self.task_encoder_initializer = params['task_encoder']['initializer']

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete'in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                    self.fixed_sigma = self.space_config['fixed_sigma']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.is_multi_discrete:
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

    def build(self, name, **kwargs):
        net = SoftModularizedA2CBuilder.Network(self.params, **kwargs)
        return net