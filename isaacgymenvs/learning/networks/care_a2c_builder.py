import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from rl_games.algos_torch import network_builder

import isaacgymenvs.learning.networks.moe_layer as moe_layer
from isaacgymenvs.learning.networks.soft_modularized_network import weight_init

from typing import List, Tuple

class CAREA2CBuilder(network_builder.NetworkBuilder):
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
            ordered_task_names = kwargs.pop('ordered_task_names')
            device = kwargs.pop('device')

            # get the dim of the real part of the obs
            true_obs_dim = obs_shape[0] - task_embedding_dim

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
        

            encoder_args = {
                'input_size' : true_obs_dim, 
                'num_experts' : self.num_experts,
                'D' : self.encoder_D, 
                'num_layers' : self.encoder_num_layers,
                'activation' : self.encoder_activation, 
                'norm_func_name' : self.encoder_normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.encoder_d2rl,
                'norm_only_first_layer' : self.encoder_norm_only_first_layer,
                'temperature' : self.temperature,
            }

            context_encoder_args = {
                'input_size' : None,                            # assigned in ContextEncoder
                'units' : self.context_encoder_units,           # hidden layer sizes
                'activation' : self.context_encoder_activation,
                'dense_func' : torch.nn.Linear,
                'context_encoder_bias' : self.context_encoder_bias,
                'path_to_load_from' : self.path_to_load_from,
                'ordered_task_names': ordered_task_names,
            }

            attention_args = {
                'input_size' : self.encoder_D,
                'units' : self.attention_units,
                'activation' : self.attention_activation,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.encoder_d2rl,
                'initializer' : self.attention_initializer,
            }

            network_args = {
                'input_size' : self.encoder_D + self.context_encoder_units[-1],
                'units' : self.care_units,
                'activation' : self.care_activation,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.encoder_d2rl,
                'initializer' : self.care_initializer,
                'multi_head' : self.multi_head,
                'unique_task_indices' : unique_task_indices,
            }
            
            if not self.separate:
                print("Building Shared Actor and Critic")
                self.actor = self._build_shared_care_network(action_dim, self.value_size, context_encoder_args, encoder_args, attention_args, network_args)
            else:
                print("Building Separate Actor")
                self.actor = self._build_care_network(action_dim, context_encoder_args, encoder_args, attention_args, network_args)
                
                print("Building Separate Critic")    
                self.critic = self._build_care_network(self.value_size, context_encoder_args, encoder_args, attention_args, network_args)
            
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
        
        def _build_care_network(self, output_dim, context_encoder_args, encoder_args, attention_args, network_args):
            return CAREPPONetwork(output_dim, context_encoder_args.copy(), encoder_args.copy(), attention_args.copy(), network_args.copy())

        def _build_shared_care_network(self, actor_output_dim, critic_output_dim, context_encoder_args, encoder_args, attention_args, network_args):
            raise NotImplementedError("Shared Actor-Critic network not implemented yet")
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            pass

        def get_default_rnn_state(self):
            pass
            
        def load(self, params):
            self.separate = params.get('separate', False)
            self.num_experts = params['encoder']['num_experts']
            self.encoder_D = params['encoder']['D']
            self.encoder_num_layers = params['encoder']['num_layers']
            self.encoder_activation = params['encoder']['activation']
            self.initializer = params['encoder']['initializer']
            self.multi_head = params['encoder']['multi_head']
            self.temperature = params['encoder']['temperature']
            self.agg_activation = params['encoder'].get('agg_activation',['relu','relu'])
            self.encoder_d2rl = params['encoder'].get('d2rl', False)
            self.encoder_norm_only_first_layer = params['encoder'].get('norm_only_first_layer', False)
            self.encoder_normalization = params.get('normalization', None)

            self.care_units = params['care']['units']
            self.care_activation = params['care']['activation']
            self.care_initializer = params['care']['initializer']
            self.care_value_activation = params.get('value_activation', 'None')

            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            self.attention_units = params['attention']['units']
            self.attention_activation = params['attention']['activation']
            self.attention_initializer = params['attention']['initializer']

            self.context_encoder_units = params['context_encoder']['units']
            self.context_encoder_activation = params['context_encoder']['activation']
            self.context_encoder_initializer = params['context_encoder']['initializer']
            self.context_encoder_bias = params['context_encoder']['bias']
            self.path_to_load_from = params['context_encoder']['path_to_load_from']

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
        net = CAREA2CBuilder.Network(self.params, **kwargs)
        return net
    

class CAREPPONetwork(network_builder.NetworkBuilder.BaseNetwork):
    def __init__(
        self,
        output_dim: int,
        context_encoder_args,
        encoder_args,
        attention_args,
        network_args,
        n_head_layers: int = 0,
        n_head_D: int = 128
    ):
        """Class to implement the CARE actor or critic network.
        """
        super().__init__()

        self.output_dim = output_dim

        self.in_features = encoder_args['input_size']
        num_experts = encoder_args['num_experts']
        num_layers = encoder_args['num_layers']
        hidden_features = encoder_args['D']
        self.temperature = encoder_args['temperature']
        
        self.unique_task_indices = network_args.pop('unique_task_indices')
        self.multi_head = network_args.pop('multi_head')
        self.network_init = network_args.pop('initializer')

        self.phi = moe_layer.FeedForward(num_experts, self.in_features, hidden_features, num_layers, hidden_features)
        for i in range(num_layers):
            if hasattr(self.phi._model[i], 'weight'):
                nn.init.xavier_uniform_(self.phi._model[i].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.phi._model[-1].weight, gain=nn.init.calculate_gain('linear'))

        self.context_encoder = ContextEncoder(context_encoder_args)

        self.attention_initializer = attention_args.pop('initializer')
        self.attention_mlp = self._build_mlp(**attention_args)
        self.attention_mlp.apply(weight_init)

        self.network = self._build_mlp(**network_args)
        self.network.apply(weight_init)

        n_contexts = len(self.unique_task_indices)
        if self.multi_head:
            self._output_heads = nn.ModuleList([])
            # multihead architecture
            for c in range(n_contexts):
                head = nn.Sequential()

                input_size = network_args['units'][-1]
                for _ in range(n_head_layers):
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
            self.head = nn.Linear(network_args['units'][-1], output_dim)
            nn.init.xavier_uniform_(self.head.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        true_obs = obs[:,:self.in_features]
        task_embedding = obs[:,self.in_features:].long()

        task_indices = torch.argmax(task_embedding, dim=1)
        # (B,) Get the task indices from one-hot embeddings

        z_context = self.context_encoder(task_indices).unsqueeze(1)
        # (B, 1, D)

        reprs = self.phi(true_obs)
        # (num_experts, B, D)
        reprs = reprs.permute(1, 2, 0)
        # (B, D, num_experts)

        # compute attention between context and encoder representations
        attn_scores = z_context.detach() @ reprs
        # (B, 1, D) @ (B, D, num_experts) -> (B, 1, num_experts)
        attn_scores = attn_scores.squeeze(1)
        # (B, num_experts)
        alphas = F.softmax(attn_scores / self.temperature, dim=1)  
        # (B, num_experts)

        # weighted sum of encoder representations
        z_enc = reprs @ alphas.unsqueeze(2)
        # (B, D, num_experts) @ (B, num_experts, 1) -> (B, D, 1)
        z_enc = z_enc.squeeze(-1)
        # (B, D)

        # pass through MLP
        z_enc = self.attention_mlp(z_enc)
        # (B, D)

        z_obs = torch.cat([z_enc, z_context.squeeze(1)], dim=1)
        # (B, 2*D)

        z_obs = self.network(z_obs)
        # (B, self.network out dim)

        if self.multi_head:
            f = torch.zeros((obs.shape[0],self.output_dim), dtype=torch.float32, device=obs.device)

            # select the task-specific head 
            for i, task_idx in enumerate(self.unique_task_indices):
                mask = task_indices == task_idx
                if mask.any(): # this batch may not have all tasks
                    f[mask] = self._output_heads[i](z_obs[mask])
        else:
            f = self.head(z_obs)
    
        return f

import json
class ContextEncoder(network_builder.NetworkBuilder.BaseNetwork):
    """
    Context encoder encodes the task embedding though a MLP.
    """
    def __init__(self, mlp_args):
        super().__init__()

        context_encoder_bias:bool = mlp_args.pop('context_encoder_bias')
        path_to_load_from = mlp_args.pop('path_to_load_from')
        ordered_task_names:List[str] = mlp_args.pop('ordered_task_names')

        with open(path_to_load_from) as f:
            metadata = json.load(f) 
        
        pretrained_embedding = torch.Tensor(
            [metadata[task] for task in ordered_task_names]
        )
        
        num_embeddings = pretrained_embedding.shape[0]
        if num_embeddings != len(ordered_task_names):
            raise ValueError(f"Number of tasks in metadata {len(ordered_task_names)} does not match number of tasks in ordered_task_names {num_embeddings} \
                             Please ensure that the metadata file has the same number of tasks as chosen tasks")
        pretrained_task_embedding_dim = pretrained_embedding.shape[1]
        pretrained_embedding = nn.Embedding.from_pretrained(
            embeddings=pretrained_embedding,
            freeze=True,
        )
        
        projection_layer = nn.Sequential(
            nn.Linear(
                in_features = pretrained_task_embedding_dim, out_features = 2 * pretrained_task_embedding_dim, bias=context_encoder_bias
            ),
            nn.ReLU(),
            nn.Linear(in_features = 2 * pretrained_task_embedding_dim, out_features = pretrained_task_embedding_dim,bias=context_encoder_bias),
            nn.ReLU(),
        )

        self.embedding = nn.Sequential(
            pretrained_embedding,
            nn.ReLU(),
            projection_layer,
        )

        mlp_args['input_size'] = pretrained_task_embedding_dim
        self.mlp = self._build_mlp(**mlp_args)
        
        # Convert existing layers to have no bias if context_encoder_bias is False and initialize weights
        for i,layer in enumerate(self.mlp):
            if isinstance(layer, nn.Linear):
                if not context_encoder_bias:
                    layer.bias = None
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, embedding):
        return self.mlp(self.embedding(embedding))