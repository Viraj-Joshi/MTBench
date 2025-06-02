import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from rl_games.algos_torch import network_builder
from isaacgymenvs.learning.networks.pq_network import QNetwork
from isaacgymenvs.learning.networks.soft_modularized_network import weight_init

from typing import List, Tuple

class SoftModularizedPQBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = SoftModularizedPQBuilder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            obs_dim = kwargs.pop('obs_dim')
            num_bins_per_dim = kwargs.pop('num_bins_per_dim')
            action_space_type = kwargs.pop('action_space_type')
            
            if 'task_indices' in kwargs:
                unique_task_indices = torch.unique(kwargs.pop('task_indices'))
                task_embedding_dim = kwargs.pop('task_embedding_dim')

                # get the dim of the real part of the obs
                true_obs_dim = obs_dim - task_embedding_dim  
            else:
                raise ValueError("Task indices not provided, use the vanilla PQAgent instead!")

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            critic_args = {
                'input_size' : true_obs_dim, 
                'num_experts' : self.num_experts,
                'D' : self.D, 
                'num_layers' : self.num_layers,
                'd2rl' : self.is_d2rl,
                'norm_first_layer_as_batchnorm' : self.norm_first_layer_as_batchnorm,
                'unique_task_indices' : unique_task_indices,
                'action_space_type' : action_space_type
            }

            task_encoder_args = {
                'input_size' : task_embedding_dim, # the in dimension of the MLP
                'units' : self.task_encoder_units, # hidden layer sizes
                'activation' : self.task_encoder_activation,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
            }

            state_encoder_args = {
                'input_size' : true_obs_dim, 
                'units' : self.state_encoder_units, 
                'activation' : self.state_encoder_activation,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
            }

            # head_args = {
            #     'units' : self.head_units,
            #     'activation' : self.head_activation,
            #     'initializer' : self.head_initializer,
            #     'output_dim': -1 # CHANGE THIS
            # }

            print("Building Critic")
            if action_space_type == "multi_discrete":
                # the output_dim is all state-action utilities 
                # for action dimensions n_a (action_dim) and discrete bins n_b (num_bins_per_dim)
                output_dim = num_bins_per_dim * actions_num
            else:
                output_dim = actions_num
            self.critic = self._build_critic(output_dim, num_bins_per_dim, task_encoder_args, state_encoder_args, critic_args)
            self.critic.apply(weight_init)

        # the critic consists of a Q network
        def _build_critic(self, output_dim, num_bins_per_dim, task_encoder_args, state_encoder_args, critic_args):
            Q = SoftModularizedQWrapper(output_dim, num_bins_per_dim, task_encoder_args, state_encoder_args, critic_args)
            return Q

        def forward(self, obs_dict):
            """TODO"""
            pass
 
        def is_separate_critic(self):
            pass

        def load(self, params):
            self.num_experts = params['q']['num_experts']
            self.D = params['q']['D']
            self.num_layers = params['q']['num_layers']
            self.activation = params['q']['activation']
            self.initializer = params['q']['initializer']
            self.is_d2rl = params['q'].get('d2rl', False)
            self.norm_first_layer_as_batchnorm = params['q'].get('norm_first_layer_as_batchnorm', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params
            self.value_shape = params.get('value_shape', 1)
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

from isaacgymenvs.learning.networks import moe_layer
from isaacgymenvs.learning.networks.soft_modularized_network import RoutingNetwork

class SoftModularizedQNetwork(torch.nn.Module):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_features: int,
        bias: bool = True,
    ):
        """Class to implement the actor/critic in
        'Multi-Task Reinforcement Learning with Soft Modularization' paper.
        It is similar to layers.FeedForward but allows selection of expert
        at each layer.
        """
        super().__init__()
        
        layers: List[nn.Module] = []
        current_in_features = hidden_features

        for _ in range(num_layers - 1):
            linear = moe_layer.Linear(
                num_experts=num_experts,
                in_features=current_in_features,
                out_features=hidden_features,
                bias=bias,
            )
            layer = nn.Sequential(linear, nn.ELU(), nn.LayerNorm(hidden_features)) # add layer_norm

            layers.append(layer)
            # Each layer is a combination of a moe layer and ReLU.
            current_in_features = hidden_features
        linear = moe_layer.Linear(
            num_experts=num_experts,
            in_features=current_in_features,
            out_features=out_features,
            bias=bias,
        )

        layers.append(linear)
        self.layers = nn.ModuleList(layers)
        self.routing_network = RoutingNetwork(
            hidden_features=hidden_features,
            num_layers=num_layers - 1,
            num_experts_per_layer=num_experts,
        )

    """
    Using Fig 2 of https://arxiv.org/pdf/2003.13661
    - f is the encoded state or state_action of shape (B,D)
    - inp IS NOT the observation
      inp IS the element wise multiplication of the encoded 
      task embedding of shape (B,D) and the encoded state f
    """
    def forward(self, f, inp: torch.Tensor) -> torch.Tensor:
        probs = self.routing_network(inp)
        # (num_layers, B, num_experts, num_experts)
        probs = probs.permute(0, 2, 3, 1)
        # (num_layers, num_experts, num_experts, B)
        num_experts = probs.shape[1]
        
        x = inp                                                                     ### IMPORTANT: using inp vs f_obs is a design choice, in the paper they use f_obs ###
        for index, layer in enumerate(self.layers[:-1]):
            p = probs[index]
            # (num_experts, num_experts, B)
            x = layer(x)                                                            # After layer transformation
            # (num_experts, B, dim2)
            _out = p.unsqueeze(-1) * x.unsqueeze(0).repeat(num_experts, 1, 1, 1)    # After multiplication with probabilities
            # (num_experts, num_experts, B, dim2)
            x = _out.sum(dim=1)                                                     # After averaging experts
            # (num_experts, batch, dim2)
        out = self.layers[-1](x).sum(dim=0)
        # (B, out_dim)
        return out
    
class SoftModularizedQWrapper(SoftModularizedQNetwork):
    def __init__(self, output_dim, num_bins_per_dim, task_encoder_args, state_encoder_args, network_args):
        num_experts = network_args['num_experts']
        in_features = network_args['input_size']
        num_layers = network_args['num_layers']
        hidden_features = network_args['D']
        
        self.action_space_type = network_args['action_space_type']  
        self.output_dim = output_dim
        self.num_bins_per_dim = num_bins_per_dim    

        super().__init__(num_experts, in_features, output_dim, num_layers, hidden_features)

        print("Building Task Encoder")
        self.task_encoder = TaskEncoder(hidden_features, **task_encoder_args)
        print("Building State Encoder")
        self.encoder = Encoder(hidden_features, **state_encoder_args)

        self.task_embedding_dim = task_encoder_args['input_size']

    def forward(self, obs):
        true_obs = obs[:,:-self.task_embedding_dim]
        # (B, obs_dim)
        task_embedding = obs[:,-self.task_embedding_dim:]
        # (B, task_embedding_dim)

        f_obs = self.encoder(true_obs)
        # (B, D)
        z = self.task_encoder(task_embedding)
        # (B, D)
        output = super().forward(f_obs, f_obs * z)
        # (B, output_dim)

        if self.action_space_type == "multi_discrete":
            # forward through Q and reshape network output of shape [B,n_a*n_b] to [B, n_a, n_b]
            action_dim = self.output_dim // self.num_bins_per_dim
            return output.view(-1,action_dim,self.num_bins_per_dim) # split to predict decoupled state-action utilities
        else:
            # just get the Q values for each action
            return output


class TaskEncoder(network_builder.NetworkBuilder.BaseNetwork):
    """
    Task encoder encodes the task embedding though fully connected layers
    """
    def __init__(self, output_dim, **mlp_args):
        super().__init__()
        if len(mlp_args['units']) == 0:
            self.mlp = nn.Sequential(nn.Linear(mlp_args['input_size'],output_dim),nn.ReLU())
        else:
            self.mlp = self._build_mlp(**mlp_args)
            last_layer = list(self.mlp.children())[-2].out_features
            self.mlp = nn.Sequential(*list(self.mlp.children()), nn.Linear(last_layer, output_dim))

    def forward(self, embedding):
        return self.mlp(embedding)

class Encoder(network_builder.NetworkBuilder.BaseNetwork):
    def __init__(self, D, **mlp_args):
        super().__init__()

        self.mlp = self._build_mlp(**mlp_args)
        last_layer = list(self.mlp.children())[-2].out_features # -2 gets the last linearity AND activation
        self.mlp = nn.Sequential(*list(self.mlp.children()), nn.Linear(last_layer, D))

    def forward(self, inp):
        return self.mlp(inp)