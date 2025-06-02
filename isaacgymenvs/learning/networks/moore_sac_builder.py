import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from rl_games.algos_torch import network_builder
from rl_games.algos_torch.sac_helper import SquashedNormal
import isaacgymenvs.learning.networks.moe_layer as moe_layer

from typing import List, Tuple

class MOORESACBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = MOORESACBuilder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            obs_shape = kwargs.pop('obs_dim')
            action_dim = kwargs.pop('action_dim')
            unique_task_indices = torch.unique(kwargs.pop('task_indices'))
            task_embedding_dim = kwargs.pop('task_embedding_dim')

            # get the dim of the real part of the obs
            obs_dim = obs_shape[0] - task_embedding_dim

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            actor_args = {
                'input_size' : obs_dim, 
                'num_experts' : self.num_experts,
                'D' : self.D, 
                'num_layers' : self.num_layers,
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer,
                'unique_task_indices' : unique_task_indices,
                'agg_activation' : self.agg_activation,
            }

            critic_args = actor_args.copy()
            critic_args['input_size'] = obs_dim + action_dim

            task_encoder_args = {
                'input_size' : task_embedding_dim, # the in dimension of the MLP
                'units' : self.task_encoder_units, # hidden layer sizes
                'activation' : self.task_encoder_activation,
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer,
                # 'task_encoder_bias': False
            }

            head_args = {
                'units' : self.head_units,
                'activation' : self.head_activation,
                'initializer' : self.head_initializer,
                'output_dim': 1
            }
            
            print("Building Actor")
            self.actor = self._build_actor(2*action_dim, task_encoder_args, self.log_std_bounds, **actor_args)

            if self.separate:
                print("Building Critic")
                self.critic = self._build_critic(1, task_encoder_args, critic_args)
                print("Building Critic Target")
                self.critic_target = self._build_critic(1, task_encoder_args, critic_args)
                self.critic_target.load_state_dict(self.critic.state_dict())
            if not self.separate:
                raise NotImplementedError("Seperate critic not implemented")
            # import ipdb; ipdb.set_trace()

        # the critic consists of double Q networks as well as the state and task encoders
        # the critic target is a copy which lags the critic AND also has its own seperate state and task encoders
        def _build_critic(self, output_dim, task_encoder_args, critic_args):
            in_features = critic_args['input_size']
            num_experts = critic_args['num_experts']
            num_layers = critic_args['num_layers']
            hidden_features = critic_args['D']
            unique_task_indices = critic_args['unique_task_indices']
            agg_activation = critic_args['agg_activation']

            Q1 = MOORESACNetwork(task_encoder_args, num_experts, in_features, hidden_features, output_dim, num_layers, hidden_features, unique_task_indices, agg_activation, is_actor=False)
            Q2 = MOORESACNetwork(task_encoder_args, num_experts, in_features, hidden_features, output_dim, num_layers, hidden_features, unique_task_indices, agg_activation, is_actor=False)

            return DoubleQCritic(Q1, Q2)

        def _build_actor(self, output_dim, task_encoder_args, log_std_bounds, **actor_args):
            in_features = actor_args['input_size']
            num_experts = actor_args['num_experts']
            num_layers = actor_args['num_layers']
            hidden_features = actor_args['D']
            unique_task_indices = actor_args['unique_task_indices']
            agg_activation = actor_args['agg_activation']

            return MOORESACNetwork(task_encoder_args, num_experts, in_features, hidden_features, output_dim, num_layers, hidden_features, unique_task_indices, agg_activation, is_actor=True, log_std_bounds=log_std_bounds)

        def forward(self, obs_dict):
            """TODO"""
            obs = obs_dict['obs']
            latent_obs = self.state_encoder(obs)
            latent_task_embedding = self.task_encoder(self.task_embedding)
            mu, sigma = self.actor(latent_obs)
            return mu, sigma
 
        def is_separate_critic(self):
            return self.separate

        def load(self, params):
            self.separate = params.get('separate', True)
            self.num_experts = params['moore']['num_experts']
            self.D = params['moore']['D']
            self.num_layers = params['moore']['num_layers']
            self.activation = params['moore']['activation']
            self.initializer = params['moore']['initializer']
            self.agg_activation = params['moore'].get('agg_activation',['relu','relu'])
            self.is_d2rl = params['moore'].get('d2rl', False)
            self.norm_only_first_layer = params['moore'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params
            self.value_shape = params.get('value_shape', 1)
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)
            self.log_std_bounds = params.get('log_std_bounds', None)

            self.head_units = params['head']['units']
            self.head_activation = params['head']['activation']
            self.head_initializer = params['head']['initializer']

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
    # _AVAILABLE_ENCODERS = {
    #     "feedforward": FeedForwardEncoder,
    # }

    """Critic network, uses 2 Q-networks to fend off overestimation"""
    def __init__(self, Q1, Q2):
        super().__init__()
        
        self.Q1 = Q1
        self.Q2 = Q2

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        real_obs = obs[:,:39] # TODO: avoid hardcoding the size of the real part of the obs
        task_embedding = obs[:,39:]
        obs_action = torch.cat([real_obs, action,task_embedding], dim=-1)

        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        return q1, q2        
    
class MOORESACNetwork(network_builder.NetworkBuilder.BaseNetwork):
    def __init__(
        self,
        task_encoder_args,
        num_experts: int,
        in_features: int,
        moe_out_features: int,
        output_dim,
        num_layers: int,
        hidden_features: int,
        unique_task_indices: torch.Tensor,
        agg_activation: List[str],
        is_actor: bool,
        log_std_bounds: Tuple[float, float] = (-10, 2),
        bias: bool = True,
        n_head_layers: int = 0,
        n_head_D: int = 256
    ):
        """Class to implement the MOORE actor or critic network.
        """
        super().__init__()

        self.in_features = in_features  
        self.output_dim = output_dim
        self.unique_task_indices = unique_task_indices
        self.is_actor = is_actor
        self.log_std_bounds = log_std_bounds
        self.agg_activation = agg_activation

        self.phi = moe_layer.FeedForward(num_experts, in_features, moe_out_features, num_layers, hidden_features, bias)
        for i in range(num_layers):
            if hasattr(self.phi._model[i], 'weight'):
                nn.init.xavier_uniform_(self.phi._model[i].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.phi._model[-1].weight, gain=nn.init.calculate_gain('linear'))

        self.orthogonal_layer = OrthogonalLayer1D()

        self.task_encoder = TaskEncoder(num_experts, **task_encoder_args)
        nn.init.xavier_uniform_(self.task_encoder.mlp.weight, gain=nn.init.calculate_gain('linear'))

        n_contexts = len(unique_task_indices)
        self.m = {}
        for i in range(n_contexts):
            self.m[self.unique_task_indices[i].item()] = i
        
        self._output_heads = nn.ModuleList([])
        # multihead architecture
        for c in range(n_contexts):
            head = nn.Sequential()
            input_size = moe_out_features
            for _ in range(n_head_layers):
                layer = nn.Linear(input_size, n_head_D)
                nn.init.xavier_uniform_(layer.weight,gain=nn.init.calculate_gain('relu'))
                head.add_module(f"head_{c}_layer_{i}",layer)

                head.add_module(f"head_{c}_act_{i}",nn.ReLU())
                input_size = n_head_D
            
            layer = nn.Linear(input_size, output_dim)
            nn.init.xavier_uniform_(layer.weight,
                                gain=nn.init.calculate_gain('linear'))
            head.add_module(f"head_{c}_out",layer)

            self._output_heads.append(head)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        real_obs = obs[:,:self.in_features]
        task_embedding = obs[:,self.in_features:]

        if (torch.argmax(task_embedding, dim=1)==0).any():
            # print(real_obs[torch.argmax(task_embedding, dim=1)==0].shape)
            raise ValueError("Invalid task_embedding of all zeros")

        if torch.isnan(real_obs).any():
            raise ValueError("NaN in real_obs")

        w = self.task_encoder(task_embedding).unsqueeze(1)
        
        # (B, 1, num_experts)
        reprs = self.phi(real_obs)
        # (num_experts, B, D)
        ortho_reprs = self.orthogonal_layer(reprs)
        # (num_experts, B, D) 
        ortho_reprs = ortho_reprs.permute(1, 0, 2)
        # (B, D, num_experts)

        # activation before aggregation
        if not self.agg_activation[0].lower() == "linear":
            ortho_reprs = getattr(F, self.agg_activation[0].lower())(ortho_reprs)
        
        ortho_reprs = w @ ortho_reprs
        # (B, 1, num_experts) @ (B, num_experts, D) -> (B, 1, D)
        ortho_reprs = ortho_reprs.squeeze(1)
        # (B, D)
        if not self.agg_activation[1].lower() == "linear":
            ortho_reprs = getattr(F, self.agg_activation[1].lower())(ortho_reprs)
        # (B, D)
        f = torch.zeros((obs.shape[0],self.output_dim), dtype=torch.float32, device=obs.device)
        
        # select the task-specific head 
        task_indices = torch.argmax(task_embedding, dim=1)  # (B,) Get the task indices from one-hot embeddings
        for i, task_idx in enumerate(self.unique_task_indices):
            mask = task_indices == task_idx
            if mask.any(): # this batch may not have all tasks
                f[mask] = self._output_heads[i](ortho_reprs[mask])
        
        # modify the output to be a distribution if this is an actor
        if self.is_actor:
            # assume a diagonal gaussian distribution
            mu, log_std = f.chunk(2, dim=-1)

            log_std_min, log_std_max = self.log_std_bounds
            log_std = torch.clamp(log_std, log_std_min, log_std_max)
            
            std = log_std.exp()

            dist = SquashedNormal(mu, std)
            return dist
        return f

    
class TaskEncoder(network_builder.NetworkBuilder.BaseNetwork):
    """
    Task encoder encodes the task embedding though a MLP.
    """
    def __init__(self, output_dim, **mlp_args):
        super().__init__()
        # if len(mlp_args['units']) == 0:
        #     self.mlp = nn.Sequential(nn.Linear(mlp_args['embedding_dim'],output_dim,nn.ReLU()))
        # else:
        #     self.mlp = self._build_mlp(**mlp_args)
        #     last_layer = list(self.mlp.children())[-2].out_features
        #     self.mlp = nn.Sequential(*list(self.mlp.children()), nn.Linear(last_layer, output_dim))
        self.mlp = torch.nn.Linear(mlp_args['input_size'], output_dim, bias=False)

    def forward(self, embedding):
        return self.mlp(embedding)

class OrthogonalLayer1D(nn.Module):

    """
        OrthogonalLayer1D make the outputs of each unit of the previous sub-layer orthogonal to each other.
        Orthogonalization is performed using Gram-Schmidt orthogonalization.
    """

    def __init__(self):
        super(OrthogonalLayer1D, self).__init__()

    def forward(self,x):

        """
        Arg:
            x: The parallel formated input with shape: [n_experts,batch_size,hidden_dim]

        return:
            basis: Orthogonalized version of the input (x). The shape of basis is [n_experts,batch_size,hidden_dim].
                   For each sample, the outputs of all of the models (n_experts) will be orthogonal
                   to each other.
        """


        x1 = torch.transpose(x, 0,1)
        basis = torch.unsqueeze(x1[:, 0, :] / (torch.unsqueeze(torch.linalg.norm(x1[:, 0, :], axis=1), 1)), 1)

        for i in range(1, x1.shape[1]):
            v = x1[:, i, :]
            v = torch.unsqueeze(v, 1)
            w = v - torch.matmul(torch.matmul(v, torch.transpose(basis, 2, 1)), basis)
            wnorm = w / (torch.unsqueeze(torch.linalg.norm(w, axis=2), 2))
            basis = torch.cat([basis, wnorm], axis=1)

        basis = torch.transpose(basis,0,1)
        return basis