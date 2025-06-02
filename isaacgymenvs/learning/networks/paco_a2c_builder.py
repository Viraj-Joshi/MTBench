import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from rl_games.algos_torch import network_builder

import isaacgymenvs.learning.networks.moe_layer as moe_layer

from typing import List, Tuple

class PACOA2CBuilder(network_builder.NetworkBuilder):
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
                'K' : self.K,
                'D' : self.D, 
                'num_layers' : self.num_layers,
                'initializer' : self.initializer,
                'activation' : self.activation, 
                'unique_task_indices' : unique_task_indices,
                'fixed_sigma' : self.fixed_sigma,
                'multi_head' : self.multi_head,
            }

            critic_args = deepcopy(actor_args)

            task_encoder_args = {
                'input_size' : task_embedding_dim, # the in dimension of the MLP
                'task_encoder_bias' : self.task_encoder_bias,
                'task_encoder_batch_norm' : self.task_encoder_batch_norm,
                'task_encoder_layer_norm' : self.task_encoder_layer_norm,
            }
            
            if not self.separate:
                print("Building Shared Actor and Critic")
                self.actor = self._build_shared_paco_network(action_dim, self.value_size, task_encoder_args, actor_args)
            else:
                print("Building Separate Actor")
                self.actor = self._build_paco_network(action_dim, task_encoder_args, actor_args)
                
                print("Building Separate Critic")    
                self.critic = self._build_paco_network(self.value_size, task_encoder_args, critic_args)
            
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
        
        def _build_paco_network(self, output_dim, task_encoder_args, network_args):
            return PACOPPONetwork(output_dim, task_encoder_args, network_args)

        def _build_shared_paco_network(self, actor_output_dim, critic_output_dim, task_encoder_args, network_args):
            return PACOSharedPPONetwork(actor_output_dim, critic_output_dim, task_encoder_args, network_args)
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            pass

        def get_default_rnn_state(self):
            pass
            
        def load(self, params):
            self.separate = params.get('separate', False)
            self.K = params['paco']['K']
            self.D = params['paco']['D']
            self.num_layers = params['paco']['num_layers']
            self.activation = params['paco']['activation']
            self.initializer = params['paco'].get('initializer',None)
            self.multi_head = params['paco']['multi_head']
            self.is_d2rl = params['paco'].get('d2rl', False)
            self.norm_only_first_layer = params['paco'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            self.task_encoder_bias = params['task_encoder']['bias']
            self.task_encoder_last_activation = params['task_encoder']['last_activation']
            self.task_encoder_batch_norm = params['task_encoder']['batch_norm']
            self.task_encoder_layer_norm = params['task_encoder']['layer_norm']

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
        net = PACOA2CBuilder.Network(self.params, **kwargs)
        return net
    

class PACOSharedPPONetwork(network_builder.NetworkBuilder.BaseNetwork):
    def __init__(
        self,
        actor_output_dim: int,
        critic_output_dim: int,
        task_encoder_args : dict,
        network_args : dict,
    ):
        """Class to implement the PACO actor and critic network.
        """
        super().__init__()

        self.actor_output_dim = actor_output_dim
        self.critic_output_dim = critic_output_dim

        self.in_features = network_args['input_size']
        K = network_args['K']
        num_layers = network_args['num_layers']
        hidden_features = network_args['D']
        self.unique_task_indices = network_args['unique_task_indices']
        self.multi_head = network_args['multi_head']
        kernel_initializer = network_args['initializer']
        # initializer for all layers of policy and value networks

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                gain=np.sqrt(1.0 / 3),
                mode='fan_in',
                distribution='uniform')
            kernel_initializer = functools.partial(nn.init.orthogonal_,gain=nn.init.calculate_gain('relu'))

        print("Creating Task Encoder")
        compositional_initializer = functools.partial(nn.init.orthogonal_, gain=nn.init.calculate_gain('linear'))
        task_encoder_args['kernel_initializer'] = kernel_initializer
        task_encoder_args['last_kernel_initializer'] = compositional_initializer
        task_encoder_args['last_activation'] = 'softmax'
        self.task_encoder = TaskEncoder(K, task_encoder_args)

        print("Creating Shared PaCO Network")

        last_kernel_initializer = functools.partial(
            nn.init.uniform_, a=-0.003, b=0.003
        )
        last_kernel_initializer = functools.partial(
            nn.init.orthogonal_,gain=nn.init.calculate_gain('linear')
        )

        self.comp_network = CompositionalEncodingNetwork(K, self.in_features, hidden_features, num_layers, hidden_features, kernel_initializer,last_kernel_initializer)
        
        # multi-head where each head is a CompositionalLayer as well
        if self.multi_head:
            n_contexts = len(self.unique_task_indices)
            self._actor_output_heads = torch.nn.ModuleList()
            self._critic_output_heads = torch.nn.ModuleList()

            for _ in range(n_contexts):
                actor_head = CompositionalLayer(K, hidden_features, actor_output_dim, activation=nn.Identity(), kernel_initializer=last_kernel_initializer)
                critic_head = CompositionalLayer(K, hidden_features, critic_output_dim, activation=nn.Identity(), kernel_initializer=last_kernel_initializer)

                self._actor_output_heads.append(actor_head)
                self._critic_output_heads.append(critic_head)

        else:
            self.actor_head = CompositionalLayer(K, hidden_features, actor_output_dim, activation=nn.Identity(), kernel_initializer=last_kernel_initializer)
            self.critic_head = CompositionalLayer(K, hidden_features, critic_output_dim, activation=nn.Identity(), kernel_initializer=last_kernel_initializer)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        true_obs = obs[:,:self.in_features]
        task_embedding = obs[:,self.in_features:]

        w = self.task_encoder(task_embedding)
        # (B, K) Get the compositional weights
        reprs = self.comp_network(true_obs, w)
        # (B, hidden_features)

        if self.multi_head: # actor and critic head for each task
            task_indices = torch.argmax(task_embedding, dim=1) # (B,) Get the task indices from one-hot embeddings
            actor_f = torch.zeros((obs.shape[0],self.actor_output_dim), dtype=torch.float32, device=obs.device)
            critic_f = torch.zeros((obs.shape[0],self.critic_output_dim), dtype=torch.float32, device=obs.device)

            for i, task_idx in enumerate(self.unique_task_indices):
                mask = task_indices == task_idx
                if mask.any():
                    actor_f[mask] = self._actor_output_heads[i](reprs[mask], w[mask])
                    critic_f[mask] = self._critic_output_heads[i](reprs[mask], w[mask])
        else: # a actor and critic head used by all tasks
            actor_f = self.actor_head(reprs, w)
            critic_f = self.critic_head(reprs, w)
        return actor_f, critic_f


class PACOPPONetwork(network_builder.NetworkBuilder.BaseNetwork):
    def __init__(
        self,
        output_dim: int,
        task_encoder_args: dict,
        network_args: dict,
    ):
        """Class to implement the PACO actor or critic network.
        """
        super().__init__()

        self.output_dim = output_dim

        self.in_features = network_args['input_size']
        K = network_args['K']
        num_layers = network_args['num_layers']
        hidden_features = network_args['D']
        self.unique_task_indices = network_args['unique_task_indices']
        self.multi_head = network_args['multi_head']
        kernel_initializer = network_args['initializer']
        # initializer for all layers of policy and value networks

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                gain=np.sqrt(1.0 / 3),
                mode='fan_in',
                distribution='uniform')
            kernel_initializer = functools.partial(nn.init.orthogonal_,gain=nn.init.calculate_gain('relu'))

        print("Creating Task Encoder")
        compositional_initializer = functools.partial(nn.init.orthogonal_, gain=nn.init.calculate_gain('linear'))
        task_encoder_args['kernel_initializer'] = kernel_initializer # not used for TaskEncoder
        task_encoder_args['last_kernel_initializer'] = compositional_initializer
        task_encoder_args['last_activation'] = 'softmax'
        self.task_encoder = TaskEncoder(K, task_encoder_args)

        
        print("Creating PaCO Network")

        last_kernel_initializer = functools.partial(
            nn.init.uniform_, a=-0.003, b=0.003
        )
        last_kernel_initializer = functools.partial(
            nn.init.orthogonal_,gain=nn.init.calculate_gain('linear')
        )

        self.comp_network = CompositionalEncodingNetwork(K, self.in_features, hidden_features, num_layers, hidden_features, kernel_initializer,last_kernel_initializer)
        
        # multi-head where each head is a CompositionalLayer as well
        if self.multi_head:
            n_contexts = len(self.unique_task_indices)
            self._output_heads = torch.nn.ModuleList()
            # multihead architecture
            for _ in range(n_contexts):
                head = CompositionalLayer(K, hidden_features, output_dim, activation=nn.Identity(), kernel_initializer=last_kernel_initializer)

                self._output_heads.append(head)
        else:
            self.head = CompositionalLayer(K, hidden_features, output_dim, activation=nn.Identity(), kernel_initializer=last_kernel_initializer)
                
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        true_obs = obs[:,:self.in_features]
        task_embedding = obs[:,self.in_features:]
        
        w = self.task_encoder(task_embedding)
        # (B,K) Get the compositional weights

        reprs = self.comp_network(true_obs, w)
        # (B,output_dim)  

        if self.multi_head:
            task_indices = torch.argmax(task_embedding, dim=1)  # (B,) Get the task indices from one-hot embeddings
            f = torch.zeros((obs.shape[0],self.output_dim), dtype=torch.float32, device=obs.device)

            # select the task-specific head 
            for i, task_idx in enumerate(self.unique_task_indices):
                mask = task_indices == task_idx
                if mask.any(): # this batch may not have all tasks
                    f[mask] = self._output_heads[i](reprs[mask], w[mask])
        else:
            f = self.head(reprs, w)
    
        return f

class TaskEncoder(torch.nn.Module):
    """
    Task encoder encodes the task embedding though a MLP.
    """
    def __init__(self, output_dim: int, mlp_args: dict):
        super().__init__()
        self.bias = mlp_args['task_encoder_bias']
        self._use_bn = mlp_args['task_encoder_batch_norm']
        self._use_ln = mlp_args['task_encoder_layer_norm']
        # self._kernel_initializer = mlp_args['kernel_initializer']
        # self._kernel_init_gain = mlp_args.get('kernel_init_gain', 1.0)
        self._last_kernel_initializer = mlp_args.get('last_kernel_initializer', None)
        last_activation = mlp_args['last_activation']
        self._bias_init_value = mlp_args.get('bias_init_value', 0.0)

        assert last_activation is not None, "Last activation must be provided"
        if last_activation == 'softmax':
            self._last_activation = nn.Softmax(dim=-1)

        if self._last_kernel_initializer is None:
            self._last_kernel_initializer = torch.nn.init.normal_
        
        # create embedding layer
        self.embedding_layer = nn.Embedding(mlp_args['input_size'], output_dim)

        self.reset_parameters()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        indices = torch.argmax(embedding, dim=-1).long()

        return self._last_activation(self.embedding_layer(indices)) # (B, output_dim)
    
    def reset_parameters(self):
        """Initialize the parameters."""
        self._last_kernel_initializer(self.embedding_layer.weight)
        
        # if self.bias:
        #     nn.init.constant_(self.last_layer.bias.data, self._bias_init_value)


class CompositionalEncodingNetwork(nn.Module):
    def __init__(
        self,
        K: int,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_features: int,
        kernel_initializer,
        last_kernel_initializer,
        bias: bool = True,
        
    ):
        """A feedforward model of compositional layers.

        Args:
            K (int): number of compositional weights.
            in_features (int): 
            out_features (int):
            num_layers (int): number of layers in the feedforward network.
            hidden_features (int): dimensionality of hidden layer in the
                feedforward network.
            kernel_initializer (Callable): initializer for the weights.
            last_kernel_initializer (Callable): initializer for the last layer
            bias (bool, optional): if set to ``False``, the layer will
            not learn an additive bias. 
        """
        super().__init__()
        self.compositional_layers = nn.ModuleList()
        
        current_in_features = in_features
        for _ in range(num_layers):
            self.compositional_layers.append(
                CompositionalLayer(
                    K=K,
                    in_features=current_in_features,
                    out_features=hidden_features,
                    bias=bias,
                    kernel_initializer=kernel_initializer,
                )
            )
            current_in_features = hidden_features
        
        if last_kernel_initializer is None:
            last_kernel_initializer = kernel_initializer
            
        # self.compositional_layers.append(
        #     CompositionalLayer(
        #         K=K,
        #         in_features=current_in_features,
        #         out_features=out_features,
        #         bias=bias,
        #         kernel_initializer=last_kernel_initializer,
        #     )
        # )

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward pass through the compositional feedforward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_features)
            w (torch.Tensor): Compositional weights of shape (B, K)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, out_features)
        """
        # Forward pass through layers
        current = x
        for layer in self.compositional_layers:
            current = layer(current, w)
        
        return current


import functools
# copied from https://github.com/TToTMooN/alf/blob/cd87d4e169db86917a7c7deb523e030dad78d7ea/alf/layers.py#L798
class CompositionalLayer(torch.nn.Module):
    def __init__(self, 
                K: int,
                in_features: int,
                out_features: int,
                activation = F.relu,
                bias: bool = True,
                use_bn: bool = False,
                use_ln: bool = False,
                kernel_initializer=None,
                bias_init_value=0.0
            ):
        """
        It maintains a set of ``K`` FC parameters for learning. During forward
        computation, it composes the set of parameters using weighted average
        with the compositional weight provided as input and then performs the
        FC computation, which is equivalent to combine the pre-activation output
        from each of the ``K`` FC layers using the compositional weight, and
        then apply normalization and activation.

        Args:
            K (int): number of weight matrices at each layer (size of compositional vector)
            in_features (int): size of each input
            out_features (int): size of each output
            activation: activation function to use
            bias (bool): whether to use bias each linear computation
            use_bn (bool): whether use Batch Normalization.
            use_ln (bool): whether use layer normalization
            kernel_initializer (Callable): initializer for the FC layer kernel.
                If none is provided a ``variance_scaling_initializer`` with gain
                as ``kernel_init_gain`` will be used.
            bias_init_value (float): a constant
        """
        super().__init__()

        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        self._activation = activation
        self._use_bn = use_bn
        self._use_ln = use_ln

        self._kernel_initializer = kernel_initializer
        self._bias_init_value = bias_init_value

        # K copies of the weight matrix
        V_hat = torch.Tensor(1, self.out_features,self.in_features)
        
        # repeat initialization for each of the K weight matrices
        self._kernel_initializer(V_hat.data[0])
        # (K, out_features, in_features)
        self.V_hat = nn.Parameter(V_hat.expand(self.K,-1,-1))

        if bias:
            self.b_hat = nn.Parameter(torch.Tensor(self.K, out_features))
            # (K, out_features)
        else:
            self.b_hat = None

        if self.b_hat is not None:
            nn.init.constant_(self.b_hat.data, self._bias_init_value)

        if use_bn:
            self._bn = nn.BatchNorm1d(out_features)
        else:
            self._bn = None
        if use_ln:
            self._ln = nn.LayerNorm(out_features)
        else:
            self._ln = None
        # self.reset_parameters()

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing y = x \cdot (\sum_i^K w_i * \hat V_i) + \sum_i^K w_i * \hat b_i

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_features)
            w (torch.Tensor): Expert weights tensor of shape (B, K)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """

        # Validate input shapes
        if x.ndim == 2:
            B = x.shape[0]
            assert w.shape[0] == B, "Batch sizes must match"
            assert w.shape[1] == self.K, f"Passed in compositional vector size {w.shape[1]} must match number of parameter sets {self.K}"

        x = x.unsqueeze(0).expand(self.K, *x.shape)
        # (K, B, in_features)

        if self.b_hat is not None:
            y = torch.baddbmm(
                    self.b_hat.unsqueeze(1), 
                    x,
                    self.V_hat.transpose(1, 2)
                )  
            # (K,1,out_features) + (K,B,in_features) @ (K,in_features,out_features) -> (K,B,out_features)
        else:
            y = torch.bmm(x, self.V_hat.transpose(1, 2))  # (K,B,out_features)
        y = y.transpose(0, 1)  # [B, K, out_features]
     
        # apply the compositional weights
        if w is not None:
            y = torch.bmm(w.unsqueeze(1), y).squeeze(1)
            # [B, 1, K] x [B, K, out_features] -> [B, 1, out_features] -> [B, out_features]
        else:
            y = y.sum(dim=1) # apply a uniform weight

        # apply normalization and activation
        if self._use_ln:
            if self.b_hat is not None:
                self._ln.bias.data.zero_()
            y = self._ln(y)
        if self._use_bn:
            if self.b_hat is not None:
                self._bn.bias.data.zero_()
            y = self._bn(y)
        y = self._activation(y)
        return y
    
    def __repr__(self):
        return super().__repr__() + f"({self.in_features}, {self.out_features}, K={self.K})"
        

    # def reset_parameters(self):
    #     """Initialize the parameters."""
    #     for i in range(self.K):
    #         if self._kernel_initializer is None:
    #             variance_scaling_init(
    #                 self.V_hat.data[i],
    #                 gain=self._kernel_init_gain,
    #                 nonlinearity=self._activation)
    #         else:
    #             self._kernel_initializer(self.V_hat.data[i])

    #     if self.b_hat is not None:
    #         nn.init.constant_(self.b_hat.data, self._bias_init_value)

    #     if self._use_ln:
    #         self._ln.reset_parameters()
    #     if self._use_bn:
    #         self._bn.reset_parameters()


    @property
    def weight(self):
        """Get the weight Tensor.

        Returns:
            Tensor: with shape (n, output_size, input_size). ``weight[i]`` is
                the weight for the i-th FC layer. ``weight[i]`` can be used for
                ``FC`` layer with the same ``input_size`` and ``output_size``
        """
        return self.V_hat

    @property
    def bias(self):
        """Get the bias Tensor.

        Returns:
            Tensor: with shape (K, output_size). ``bias[i]`` is the bias for the
                i-th FC layer. ``bias[i]`` can be used for ``FC`` layer with
                the same ``input_size`` and ``output_size``
        """
        return self.b_hat

def variance_scaling_init(tensor,
                          gain=1.0,
                          mode="fan_in",
                          distribution="truncated_normal",
                          calc_gain_after_activation=True,
                          nonlinearity='relu',
                          transposed=False):
        """Implements TensorFlow's `VarianceScaling` initializer.

        `<https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/ops/init_ops.py#L437>`_

        A potential benefit of this intializer is that we can sample from a truncated
        normal distribution: ``scipy.stats.truncnorm(a=-2, b=2, loc=0., scale=1.)``.

        Also incorporates PyTorch's calculation of the recommended gains that taking
        nonlinear activations into account, so that after N layers, the final output
        std (in linear space) will be a constant regardless of N's value (when N is
        large). This auto gain probably won't make much of a difference if the
        network is shallow, as in most RL cases.

        Example usage:

        .. code-block:: python

            from alf.networks.initializers import variance_scaling_init
            layer = nn.Linear(2, 2)
            variance_scaling_init(layer.weight.data,
                                nonlinearity=nn.functional.leaky_relu)
            nn.init.zeros_(layer.bias.data)

        Args:
            tensor (torch.Tensor): the weights to be initialized
            gain (float): a positive scaling factor for weight std. Different from
                tf's implementation, this number is applied outside of ``math.sqrt``.
                Note that if ``calc_gain_after_activation=True``, this number will be
                an additional gain factor on top of that.
            mode (str): one of "fan_in", "fan_out", and "fan_avg"
            distribution (str): one of "uniform", "untruncated_normal" and
                "truncated_normal". If the latter, the weights will be sampled
                from a normal distribution truncated at ``(-2, 2)``.
            calc_gain_after_activation (bool): whether automatically calculate the
                std gain of applying nonlinearity after this layer. A nonlinear
                activation (e.g., relu) might change std after the transformation,
                so we need to compensate for that. Only used when mode=="fan_in".
            nonlinearity (Callable): any callable activation function
            transposed (bool): a flag indicating if the weight tensor has been
                tranposed (e.g., ``nn.ConvTranspose2d``). In that case, `fan_in` and
                `fan_out` should be swapped.

        Returns:
            torch.Tensor: a randomly initialized weight tensor
        """

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        if transposed:
            fan_in, fan_out = fan_out, fan_in

        assert mode in ["fan_in", "fan_out", "fan_avg"], \
            "Unrecognized mode %s!" % mode
        if mode == "fan_in":
            size = max(1.0, fan_in)
        elif mode == "fan_out":
            size = max(1.0, fan_out)
        else:
            size = max(1.0, (fan_in + fan_out) / 2.0)

        if (calc_gain_after_activation and mode == "fan_in"):
            gain *= nn.init.calculate_gain(nonlinearity)

        std = gain / np.sqrt(size)
        if distribution == "truncated_normal":
            scale = 0.87962566  # scipy.stats.truncnorm.std(-2.0, 2.0)
            std /= scale
            nn.init.trunc_normal_(tensor, a=-2.0, b=2.0)  # truncate within 2 std
            return tensor.mul_(std)
        elif distribution == "uniform":
            limit = np.sqrt(3.0) * std
            with torch.no_grad():
                return tensor.uniform_(-limit, limit)
        elif distribution == "untruncated_normal":
            with torch.no_grad():
                return tensor.normal_(0, std)
        else:
            raise ValueError("Invalid `distribution` argument:", distribution)
        