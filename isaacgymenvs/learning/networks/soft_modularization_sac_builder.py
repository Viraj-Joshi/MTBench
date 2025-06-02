from rl_games.algos_torch import network_builder
from rl_games.algos_torch.sac_helper import SquashedNormal
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from isaacgymenvs.learning.networks import moe_layer

from isaacgymenvs.learning.networks.soft_modularized_network import SoftModularizedMLP, weight_init
        

class SoftModularizedSACBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = SoftModularizedSACBuilder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            obs_dim = kwargs.pop('obs_dim')
            action_dim = kwargs.pop('action_dim')
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            task_indices = kwargs.pop('task_indices')
            task_embedding_dim = kwargs.pop('task_embedding_dim')

            # get the dim of the real part of the obs
            true_obs_dim = obs_dim - task_embedding_dim

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            actor_args = {
                'input_size' : self.D, 
                'num_experts' : self.num_experts,
                'D' : self.D, 
                'num_layers' : self.soft_num_layers,
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }

            critic_args = actor_args.copy()
            # critic_args['input_size'] = obs_dim + action_dim

            state_encoder_args = {
                'input_size' : true_obs_dim, 
                'units' : self.state_encoder_units, 
                'activation' : self.state_encoder_activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer,
                'encoder_bias' : True
            }

            state_action_encoder_args = {
                'input_size' : true_obs_dim + action_dim, 
                'units' : self.state_encoder_units, 
                'activation' : self.state_encoder_activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer,
                'encoder_bias' : True
            }
            
            task_encoder_args = {
                'input_size' : task_embedding_dim, # the in dimension of the MLP
                'units' : self.task_encoder_units, # hidden layer sizes
                'activation' : self.task_encoder_activation,
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer,
                'encoder_bias' : True
            }
    
            print("Building Actor")
            self.actor = self._build_actor(2*action_dim, state_encoder_args, task_encoder_args, self.log_std_bounds, actor_args)

            if self.separate:
                print("Building Critic")
                self.critic = self._build_critic(1, state_action_encoder_args, task_encoder_args, critic_args)
                print("Building Critic Target")
                self.critic_target = self._build_critic(1, state_action_encoder_args, task_encoder_args, critic_args)
                self.critic_target.load_state_dict(self.critic.state_dict())
            else:
                raise NotImplementedError("Shared actor critic is not supported option")
                
        # the critic consists of double Q networks as well as the state and task encoders
        # the critic target is a copy of Q which lags the critic Q AND also has its own lagging state and task encoders
        def _build_critic(self, output_dim, state_action_encoder_args, task_encoder_args, critic_args):
            return DoubleQCritic(output_dim, task_encoder_args, state_action_encoder_args, critic_args)

        # the actor is a soft modularized mlp with the state and task encoders
        def _build_actor(self, output_dim, state_encoder_args, task_encoder_args, log_std_bounds, actor_args):
            return SoftModularizedTanhDiagGaussianActor(output_dim, task_encoder_args, state_encoder_args, actor_args, log_std_bounds)

        def forward(self, obs_dict):
            """TODO"""
            pass
 
        def is_separate_critic(self):
            return self.separate

        def load(self, params):
            self.separate = params.get('separate', True)
            self.num_experts = params['soft_network']['num_experts']
            self.D = params['soft_network']['D']
            self.soft_num_layers = params['soft_network']['num_layers']
            self.activation = params['soft_network']['activation']
            self.initializer = params['soft_network']['initializer']
            self.is_d2rl = params['soft_network'].get('d2rl', False)
            self.norm_only_first_layer = params['soft_network'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params
            self.value_shape = params.get('value_shape', 1)
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)
            self.log_std_bounds = params.get('log_std_bounds', None)

            self.state_encoder_units = params['state_encoder']['units']
            self.state_encoder_activation = params['state_encoder']['activation']
            self.state_encoder_initializer = params['state_encoder']['initializer']

            self.task_encoder_units = params['task_encoder']['units']
            self.task_encoder_activation = params['task_encoder']['activation']
            self.task_encoder_initializer = params['task_encoder']['initializer']

            if self.has_space:
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False


class DoubleQCritic(network_builder.NetworkBuilder.BaseNetwork):
    """Critic network, uses 2 Q-networks to fend off overestimation"""
    def __init__(self, output_dim, task_encoder_args, state_action_encoder_args, critic_args):
        super().__init__()

        num_experts = critic_args['num_experts']
        in_features = critic_args['input_size']
        num_layers = critic_args['num_layers']
        hidden_features = critic_args['D']
        
        print("Building soft-modularized Q Networks")
        self.Q1 = SoftModularizedMLP(num_experts, in_features, output_dim, num_layers, hidden_features)
        self.Q2 = SoftModularizedMLP(num_experts, in_features, output_dim, num_layers, hidden_features)
        self.Q1.apply(weight_init)
        self.Q2.apply(weight_init)

        print("Building Task Encoders")
        self.task_encoder_1 = Encoder(hidden_features, task_encoder_args)
        self.task_encoder_2 = Encoder(hidden_features, task_encoder_args)

        print("Building State Action Encoders")
        self.state_action_encoder_1 = Encoder(hidden_features, state_action_encoder_args)
        self.state_action_encoder_2 = Encoder(hidden_features, state_action_encoder_args)

        self.task_embedding_dim = task_encoder_args['input_size']

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        true_obs = obs[:,:-self.task_embedding_dim]
        # (B, true_obs_dim)
        task_embedding = obs[:,-self.task_embedding_dim:]
        obs_action = torch.cat([true_obs, action], dim=-1)
        # (B, true_obs_dim + action_dim)

        encoded_obs_action_1 = self.state_action_encoder_1(obs_action)
        # (B, D)
        encoded_task_embedding_1 = self.task_encoder_1(task_embedding)
        # (B, D)

        encoded_obs_action_2 = self.state_action_encoder_2(obs_action)
        # (B, D)
        encoded_task_embedding_2 = self.task_encoder_2(task_embedding)
        # (B, D)
        
        q1 = self.Q1(encoded_obs_action_1, encoded_obs_action_1 * encoded_task_embedding_1)
        # (B, output_dim=1)
        q2 = self.Q2(encoded_obs_action_2, encoded_obs_action_2 * encoded_task_embedding_2)
        # (B, output_dim=1)

        return q1, q2
       
class SoftModularizedTanhDiagGaussianActor(SoftModularizedMLP):
    def __init__(self, output_dim, task_encoder_args, state_encoder_args, actor_args, log_std_bounds, should_use_multi_head_policy=False):
        in_features = actor_args['input_size']
        num_experts = actor_args['num_experts']
        num_layers = actor_args['num_layers']
        hidden_features = actor_args['D']
        
        
        self.should_use_multi_head_policy = should_use_multi_head_policy
        if self.should_use_multi_head_policy:
            self.heads = self._make_head(
                            input_dim=hidden_features,
                            output_dim=output_dim,
                            hidden_dim=hidden_features,
                            num_layers=2,
                        )
        super().__init__(num_experts, in_features, output_dim, num_layers, hidden_features)
        
        self.layers.apply(weight_init) # init actor network
        self.routing_network.apply(weight_init) # init routing network

        print("Building Task Encoder for Actor")
        self.task_encoder = Encoder(hidden_features, task_encoder_args)

        print("Building State Encoder for Actor")
        self.state_encoder = Encoder(hidden_features, state_encoder_args)

        self.log_std_bounds = log_std_bounds
        self.task_embedding_dim = task_encoder_args['input_size']

    # following method copied from MTRL library, not tested
    def _make_head(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> torch.nn.Module:
        """Make the heads of the actor

        Args:
            input_dim (int): size of the input.
            hidden_dim (int): size of the hidden layer of the head.
            num_layers (int): number of layers in the model.
        Returns:
            ModelType:
        """
        return moe_layer.FeedForward(
            num_experts=10, # TODO: make this a parameter
            in_features=input_dim,
            out_features=output_dim,
            hidden_features=hidden_dim,
            num_layers=num_layers,
            bias=True,
        )
        

    def forward(self, inp):
        obs = inp[:,:-self.task_embedding_dim]
        # (B, obs_dim)
        task_embedding = inp[:,-self.task_embedding_dim:]
        # (B, task_embedding_dim)

        f_obs = self.state_encoder(obs)
        z = self.task_encoder(task_embedding)
        output = super().forward(f_obs, f_obs * z)

        if self.should_use_multi_head_policy:
            output = self.heads(F.relu(output))

        # assume a diagonal gaussian distribution
        mu, log_std = output.chunk(2, dim=-1)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clamp(log_std, log_std_min, log_std_max)
        
        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist
    
class Encoder(network_builder.NetworkBuilder.BaseNetwork):
    """
    Encoder encodes input though a MLP.
    """
    def __init__(self, output_dim, mlp_args):
        super().__init__()
        encoder_bias = mlp_args['encoder_bias']
        if len(mlp_args['units']) == 0:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_args['input_size'], output_dim, bias=encoder_bias)
            )
            nn.init.xavier_uniform_(self.mlp[0].weight, gain=nn.init.calculate_gain('linear'))
        else:
            build_args = mlp_args.copy()
            build_args.pop('encoder_bias')
            self.mlp = self._build_mlp(**build_args)
            last_layer_dim = list(self.mlp.children())[-2].out_features
            
            # Convert existing layers to have no bias if encoder_bias is False and initialize weights
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    if not encoder_bias:
                        layer.bias = None
                    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            
            # create final layer
            final_layer = nn.Linear(last_layer_dim, output_dim, bias=encoder_bias)
            nn.init.xavier_uniform_(final_layer.weight, gain=nn.init.calculate_gain('linear'))
            
            # Add final layer
            self.mlp = nn.Sequential(
                *list(self.mlp.children()), 
                final_layer
            )
                
    def forward(self, inp):
        return self.mlp(inp)