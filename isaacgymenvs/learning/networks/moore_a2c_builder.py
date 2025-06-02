import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from rl_games.algos_torch import network_builder

import isaacgymenvs.learning.networks.moe_layer as moe_layer

from typing import List, Tuple

class MOOREA2CBuilder(network_builder.NetworkBuilder):
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
            device = kwargs.pop('device')

            # get the dim of the real part of the obs
            true_obs_dim = obs_shape[0] - task_embedding_dim

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
        

            actor_args = {
                'input_size' : true_obs_dim, 
                'num_experts' : self.num_experts,
                'D' : self.D, 
                'num_layers' : self.num_layers,
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer,
                'unique_task_indices' : unique_task_indices,
                'fixed_sigma' : self.fixed_sigma,
                'multi_head' : self.multi_head,
                'agg_activation' : self.agg_activation,
            }

            task_encoder_args = {
                'input_size' : task_embedding_dim, # the in dimension of the MLP
                'units' : self.task_encoder_units, # hidden layer sizes
                'activation' : self.task_encoder_activation,
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer,
                'task_encoder_bias' : self.task_encoder_bias
            }

            head_args = {
                'units' : self.head_units,
                'activation' : self.head_activation,
                'initializer' : self.head_initializer,
                'output_dim': 1
            }
            
            if not self.separate:
                print("Building Shared Actor and Critic")
                self.actor = self._build_shared_moore_network(action_dim, self.value_size, task_encoder_args, **actor_args)
            else:
                print("Building Separate Actor")
                self.actor = self._build_moore_network(action_dim, task_encoder_args.copy(), actor_args, head_args)
                
                print("Building Separate Critic")    
                self.critic = self._build_moore_network(self.value_size, task_encoder_args.copy(), actor_args, head_args)
            
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
        
        def _build_moore_network(self, output_dim, task_encoder_args, network_args, head_args):
            return MOOREPPONetwork(output_dim, task_encoder_args, network_args, head_args)

        def _build_shared_moore_network(self, actor_output_dim, critic_output_dim, task_encoder_args, **network_args):
            in_features = network_args['input_size']
            num_experts = network_args['num_experts']
            num_layers = network_args['num_layers']
            hidden_features = network_args['D']
            unique_task_indices = network_args['unique_task_indices']
            multi_head = network_args['multi_head']
            agg_activation = network_args['agg_activation']
            task_encoder_bias = network_args['task_encoder_bias']
            fixed_sigma = network_args['fixed_sigma']

            return MOORESharedPPONetwork(actor_output_dim, critic_output_dim, task_encoder_args, task_encoder_bias, num_experts, in_features, num_layers, hidden_features, unique_task_indices, multi_head, agg_activation, fixed_sigma)
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            pass

        def get_default_rnn_state(self):
            pass
            
        def load(self, params):
            self.separate = params.get('separate', False)
            self.num_experts = params['moore']['num_experts']
            self.D = params['moore']['D']
            self.num_layers = params['moore']['num_layers']
            self.activation = params['moore']['activation']
            self.initializer = params['moore']['initializer']
            self.multi_head = params['moore']['multi_head']
            self.agg_activation = params['moore'].get('agg_activation',['relu','relu'])
            self.is_d2rl = params['moore'].get('d2rl', False)
            self.norm_only_first_layer = params['moore'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            self.head_units = params['head']['units']
            self.head_activation = params['head']['activation']
            self.head_initializer = params['head']['initializer']

            self.task_encoder_units = params['task_encoder']['units']
            self.task_encoder_activation = params['task_encoder']['activation']
            self.task_encoder_initializer = params['task_encoder']['initializer']
            self.task_encoder_bias = params['task_encoder']['bias']

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
        net = MOOREA2CBuilder.Network(self.params, **kwargs)
        return net
    

class MOOREPPONetwork(network_builder.NetworkBuilder.BaseNetwork):
    def __init__(
        self,
        output_dim: int,
        task_encoder_args,
        network_args,
        head_args,
        n_head_layers: int = 0,
        n_head_D: int = 256
    ):
        """Class to implement the MOORE actor or critic network.
        """
        super().__init__()

        self.output_dim = output_dim

        self.in_features = network_args['input_size']
        num_experts = network_args['num_experts']
        num_layers = network_args['num_layers']
        hidden_features = network_args['D']
        self.unique_task_indices = network_args['unique_task_indices']
        self.multi_head = network_args['multi_head']
        self.agg_activation = network_args['agg_activation']
        fixed_sigma = network_args['fixed_sigma']

        self.phi = moe_layer.FeedForward(num_experts, self.in_features, hidden_features, num_layers, hidden_features)
        for i in range(num_layers):
            if hasattr(self.phi._model[i], 'weight'):
                nn.init.xavier_uniform_(self.phi._model[i].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.phi._model[-1].weight, gain=nn.init.calculate_gain('linear'))

        self.orthogonal_layer = OrthogonalLayer1D()

        self.task_encoder = TaskEncoder(num_experts, task_encoder_args)

        
        if self.multi_head:
            n_contexts = len(self.unique_task_indices)
            self._output_heads = nn.ModuleList([])
            # multihead architecture
            for c in range(n_contexts):
                head = nn.Sequential()

                input_size = hidden_features
                for i in range(n_head_layers):
                    layer = nn.Linear(input_size, n_head_D)
                    nn.init.xavier_uniform_(layer.weight,gain=nn.init.calculate_gain('relu'))
                    head.add_module(f"head_{c}_layer_{i}",layer)
                    head.add_module(f"head_{c}_act_{i}",nn.ReLU())
                    
                    input_size = n_head_D
                
                output_layer = nn.Linear(input_size, output_dim)
                nn.init.xavier_uniform_(output_layer.weight,
                                    gain=nn.init.calculate_gain('linear'))
                head.add_module(f"head_{c}_out",output_layer)
                
                self._output_heads.append(head)
        else:
            self.head = nn.Linear(hidden_features + task_encoder_args['input_size'], output_dim)
            nn.init.xavier_uniform_(self.head.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        real_obs = obs[:,:self.in_features]
        task_embedding = obs[:,self.in_features:]

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
        
        if self.multi_head:
            f = torch.zeros((obs.shape[0],self.output_dim), dtype=torch.float32, device=obs.device)

            # select the task-specific head 
            task_indices = torch.argmax(task_embedding, dim=1)  # (B,) Get the task indices from one-hot embeddings
            for i, task_idx in enumerate(self.unique_task_indices):
                mask = task_indices == task_idx
                if mask.any(): # this batch may not have all tasks
                    f[mask] = self._output_heads[i](ortho_reprs[mask])
        else:
            """
            in the single head case,
            we condition the output head on the context by concatenating the context (one-hot task embedding) to the output of the representation block
            """ 
            inp = torch.cat([ortho_reprs, task_embedding.float()], dim=1)
            f = self.head(inp)
    
        return f
    
class MOORESharedPPONetwork(network_builder.NetworkBuilder.BaseNetwork):
    def __init__(
        self,
        actor_output_dim: int,
        critic_output_dim: int,
        task_encoder_args,
        task_encoder_bias: bool,
        num_experts: int,
        in_features: int,
        num_layers: int,
        hidden_features: int,
        unique_task_indices: torch.Tensor,
        multi_head: bool,
        agg_activation: List[str],
        fixed_sigma: bool,
        n_head_layers: int = 0,
        n_head_D: int = 256
    ):
        """Class to implement the MOORE actor or critic network.
        """
        super().__init__()

        self.in_features = in_features  
        self.actor_output_dim = actor_output_dim
        self.critic_output_dim = critic_output_dim
        self.unique_task_indices = unique_task_indices
        self.agg_activation = agg_activation
        self.multi_head = multi_head

        self.phi = moe_layer.FeedForward(num_experts, in_features, hidden_features, num_layers, hidden_features)
        for i in range(num_layers):
            if hasattr(self.phi._model[i], 'weight'):
                nn.init.xavier_uniform_(self.phi._model[i].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.phi._model[-1].weight, gain=nn.init.calculate_gain('linear'))

        self.orthogonal_layer = OrthogonalLayer1D()

        self.task_encoder = TaskEncoder(num_experts, task_encoder_args, task_encoder_bias)

        n_contexts = len(unique_task_indices)
        self.m = {}
        for i in range(n_contexts):
            self.m[self.unique_task_indices[i].item()] = i
        
        if self.multi_head:
            self._actor_output_heads = nn.ModuleList([])
            self._critic_output_heads = nn.ModuleList([])
            # multihead architecture
            for c in range(n_contexts):
                actor_head = nn.Sequential()
                critic_head = nn.Sequential()

                input_size = hidden_features
                for _ in range(n_head_layers):
                    actor_layer = nn.Linear(input_size, n_head_D)
                    nn.init.xavier_uniform_(actor_layer.weight,gain=nn.init.calculate_gain('relu'))
                    actor_head.add_module(f"head_{c}_actor_layer_{i}",actor_layer)
                    actor_head.add_module(f"head_{c}_actor_act_{i}",nn.ReLU())

                    critic_layer = nn.Linear(input_size, n_head_D)
                    nn.init.xavier_uniform_(critic_layer.weight,gain=nn.init.calculate_gain('relu'))
                    critic_head.add_module(f"head_{c}_critic_layer_{i}",critic_layer)
                    critic_head.add_module(f"head_{c}_critic_act_{i}",nn.ReLU())
                    
                    input_size = n_head_D
                
                actor_output_layer = nn.Linear(input_size, actor_output_dim)
                nn.init.xavier_uniform_(actor_output_layer.weight,
                                    gain=nn.init.calculate_gain('linear'))
                actor_head.add_module(f"head_{c}_actor_out",actor_output_layer)

                critic_output_layer = nn.Linear(input_size, critic_output_dim)
                nn.init.xavier_uniform_(critic_output_layer.weight,
                                    gain=nn.init.calculate_gain('linear'))
                critic_head.add_module(f"head_{c}_critic_out",critic_output_layer)
                
                self._critic_output_heads.append(critic_head)
                self._actor_output_heads.append(actor_head)
        else:
            self.actor_head = nn.Linear(hidden_features + task_encoder_args['input_size'], actor_output_dim)
            nn.init.xavier_uniform_(self.actor_head.weight, gain=nn.init.calculate_gain('linear'))
            
            # single head architecture
            self.critic_head = nn.Linear(hidden_features + task_encoder_args['input_size'], critic_output_dim)
            nn.init.xavier_uniform_(self.critic_head.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        real_obs = obs[:,:self.in_features]
        task_embedding = obs[:,self.in_features:]

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
        
        if self.multi_head:
            f_mu = torch.zeros((obs.shape[0],self.actor_output_dim), dtype=torch.float32, device=obs.device)
            f_values = torch.zeros((obs.shape[0],self.critic_output_dim), dtype=torch.float32, device=obs.device)

            # select the task-specific head 
            task_indices = torch.argmax(task_embedding, dim=1)  # (B,) Get the task indices from one-hot embeddings
            for i, task_idx in enumerate(self.unique_task_indices):
                mask = task_indices == task_idx
                if mask.any(): # this batch may not have all tasks
                    f_mu[mask] = self._actor_output_heads[i](ortho_reprs[mask])
                    f_values[mask] = self._critic_output_heads[i](ortho_reprs[mask])
        else:
            """
            in the single head case,
            we condition the output head on the context by concatenating the context (one-hot task embedding) to the output of the representation block
            """ 
            inp = torch.cat([ortho_reprs, task_embedding], dim=1)
            f_mu = self.actor_head(inp)
            f_values = self.critic_head(inp)
    
        return f_mu, f_values

    
class TaskEncoder(network_builder.NetworkBuilder.BaseNetwork):
    """
    Task encoder encodes the task embedding though a MLP.
    """
    def __init__(self, output_dim, mlp_args):
        super().__init__()
        task_encoder_bias = mlp_args.pop('task_encoder_bias')
        if len(mlp_args['units']) == 0:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_args['input_size'], output_dim, bias=task_encoder_bias)
            )
            nn.init.xavier_uniform_(self.mlp[0].weight, gain=nn.init.calculate_gain('linear'))
        else:
            self.mlp = self._build_mlp(**mlp_args)
            last_layer_dim = list(self.mlp.children())[-2].out_features
            
            # Convert existing layers to have no bias if task_encoder_bias is False and initialize weights
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    if not task_encoder_bias:
                        layer.bias = None
                    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            
            # create final layer
            final_layer = nn.Linear(last_layer_dim, output_dim, bias=task_encoder_bias)
            nn.init.xavier_uniform_(final_layer.weight, gain=nn.init.calculate_gain('linear'))
            
            # Add final layer
            self.mlp = nn.Sequential(
                *list(self.mlp.children()), 
                final_layer
            )
                

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