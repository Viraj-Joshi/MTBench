from torch import nn
import torch

from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder


class AsymmetricA2CBuilder(A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):

        def parse_params(self, params):
            self.separate = params.get('separate', False)
            self.multi_head = params.get('multi_head', False)
            
            # Actor parameters
            self.actor_units = params['actor']['units']
            self.actor_activation = params['actor']['activation']
            self.actor_initializer = params['actor']['initializer']
            self.actor_is_d2rl = params['actor'].get('d2rl', False)
            self.actor_norm_only_first_layer = params['actor'].get('norm_only_first_layer', False)
            self.actor_normalization = params['actor'].get('normalization', None)

            # Critic parameters
            self.critic_units = params['critic']['units']
            self.critic_activation = params['critic']['activation']
            self.critic_initializer = params['critic']['initializer']
            self.critic_is_d2rl = params['critic'].get('d2rl', False)
            self.critic_norm_only_first_layer = params['critic'].get('norm_only_first_layer', False)
            self.critic_normalization = params['critic'].get('normalization', None)
            self.value_activation = params['critic'].get('value_activation', 'None')

            # Optional RNN (e.g. SAPG-style LSTM), configured under `critic.rnn`.
            #   separate=True  -> critic-only RNN; the actor stays feedforward and only the
            #                     value function is recurrent.
            #   separate=False -> shared recurrent trunk (one network for both heads):
            #                     obs -> LSTM -> MLP -> action/value, exactly as SAPG does.
            critic_rnn = params['critic'].get('rnn', None)
            self.has_rnn = critic_rnn is not None
            self.has_critic_rnn = self.has_rnn and self.separate
            self.has_shared_rnn = self.has_rnn and not self.separate
            if self.has_rnn:
                self.critic_rnn_name = critic_rnn['name']
                self.critic_rnn_units = critic_rnn['units']
                self.critic_rnn_layers = critic_rnn.get('layers', 1)
                self.critic_rnn_ln = critic_rnn.get('layer_norm', False)
                self.critic_rnn_before_mlp = critic_rnn.get('before_mlp', False)

            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete' in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous' in params['space']
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

        def __init__(self, params, **kwargs):
            self.actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = kwargs.pop('num_seqs', 1)

            learn_task_embedding = kwargs.pop('learn_task_embedding', False)
            self.task_indices = kwargs['task_indices']
            self.num_tasks = torch.unique(self.task_indices).shape[0]

            task_embedding_dim = kwargs.pop('task_embedding_dim')

            num_policies = kwargs.pop('num_policies', 0)
            policy_embedding_dim = kwargs.pop('policy_embedding_dim', 0)
            learn_policy_embedding = kwargs.pop(
                'learn_policy_embedding', num_policies > 1 and policy_embedding_dim > 0
            )

            real_obs_dim = input_shape[0] - self.num_tasks - num_policies
            task_out = task_embedding_dim if learn_task_embedding else self.num_tasks
            if num_policies > 0:
                pol_out = policy_embedding_dim if learn_policy_embedding else num_policies
            else:
                pol_out = 0
            obs_dim = real_obs_dim + task_out + pol_out
            
            NetworkBuilder.BaseNetwork.__init__(self)
            self.parse_params(params)

            self.num_policies = num_policies

            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            mlp_input_size = obs_dim

            # Path: MLP
            actor_out_size = self.actor_units[-1] if len(self.actor_units) > 0 else mlp_input_size
            critic_out_size = self.critic_units[-1] if len(self.critic_units) > 0 else mlp_input_size

            # If not separate, critic output size is same as actor
            if not self.separate:
                critic_out_size = actor_out_size
                
            actor_mlp_args = {
                'input_size' : mlp_input_size, 
                'units' : self.actor_units, 
                'activation' : self.actor_activation,
                'norm_func_name' : self.actor_normalization, 
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.actor_is_d2rl, 
                'norm_only_first_layer' : self.actor_norm_only_first_layer,
            }
            task_embedding_args = {
                'learn_task_embedding' : learn_task_embedding,
                'task_embedding_dim' : task_embedding_dim,
                'num_tasks' : self.num_tasks
            }
            policy_embedding_args = {
                'num_policies' : num_policies,
                'embedding_dim' : policy_embedding_dim,
                'num_tasks' : self.num_tasks,
                'learn' : learn_policy_embedding,
            }
            if self.has_shared_rnn:
                # Shared recurrent trunk (separate=False): obs -> embedders -> LSTM -> MLP,
                # whose output feeds BOTH the action heads and the value head (SAPG style).
                self.actor_policy_embedder = PolicyEmbedder(**policy_embedding_args)
                self.actor_task_embedder = TaskEmbedder(task_embedding_args)
                shared_mlp_common = {
                    'activation': self.actor_activation,
                    'norm_func_name': self.actor_normalization,
                    'dense_func': torch.nn.Linear,
                    'd2rl': self.actor_is_d2rl,
                    'norm_only_first_layer': self.actor_norm_only_first_layer,
                }
                if self.critic_rnn_before_mlp:
                    # obs -> LSTM -> MLP -> heads  (SAPG ordering)
                    self.a_rnn = self._build_rnn(
                        self.critic_rnn_name, mlp_input_size, self.critic_rnn_units, self.critic_rnn_layers
                    )
                    if self.critic_rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.critic_rnn_units)
                    self.actor_mlp_net = self._build_mlp(
                        input_size=self.critic_rnn_units, units=self.actor_units, **shared_mlp_common
                    )
                    actor_out_size = self.actor_units[-1] if len(self.actor_units) > 0 else self.critic_rnn_units
                else:
                    # obs -> MLP -> LSTM -> heads
                    self.actor_mlp_net = self._build_mlp(
                        input_size=mlp_input_size, units=self.actor_units, **shared_mlp_common
                    )
                    rnn_in = self.actor_units[-1] if len(self.actor_units) > 0 else mlp_input_size
                    self.a_rnn = self._build_rnn(
                        self.critic_rnn_name, rnn_in, self.critic_rnn_units, self.critic_rnn_layers
                    )
                    if self.critic_rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.critic_rnn_units)
                    actor_out_size = self.critic_rnn_units
                critic_out_size = actor_out_size  # one shared trunk feeds both heads
            else:
                self.actor_mlp = self._build_actor(actor_mlp_args, task_embedding_args, policy_embedding_args)

            if self.separate:
                # critic embedders (task / policy one-hot -> learned embedding), applied before
                # the optional RNN so the recurrent state sees the embedded observation
                self.critic_policy_embedder = PolicyEmbedder(**policy_embedding_args)
                self.critic_task_embedder = TaskEmbedder(task_embedding_args)
                critic_mlp_common = {
                    'activation': self.critic_activation,
                    'norm_func_name': self.critic_normalization,
                    'dense_func': torch.nn.Linear,
                    'd2rl': self.critic_is_d2rl,
                    'norm_only_first_layer': self.critic_norm_only_first_layer,
                }
                if self.has_critic_rnn and self.critic_rnn_before_mlp:
                    # obs -> RNN -> MLP -> value  (SAPG ordering)
                    self.c_rnn = self._build_rnn(
                        self.critic_rnn_name, mlp_input_size, self.critic_rnn_units, self.critic_rnn_layers
                    )
                    if self.critic_rnn_ln:
                        self.c_layer_norm = torch.nn.LayerNorm(self.critic_rnn_units)
                    self.critic_mlp_net = self._build_mlp(
                        input_size=self.critic_rnn_units, units=self.critic_units, **critic_mlp_common
                    )
                    critic_out_size = self.critic_units[-1] if len(self.critic_units) > 0 else self.critic_rnn_units
                elif self.has_critic_rnn and not self.critic_rnn_before_mlp:
                    # obs -> MLP -> RNN -> value
                    self.critic_mlp_net = self._build_mlp(
                        input_size=mlp_input_size, units=self.critic_units, **critic_mlp_common
                    )
                    rnn_in = self.critic_units[-1] if len(self.critic_units) > 0 else mlp_input_size
                    self.c_rnn = self._build_rnn(
                        self.critic_rnn_name, rnn_in, self.critic_rnn_units, self.critic_rnn_layers
                    )
                    if self.critic_rnn_ln:
                        self.c_layer_norm = torch.nn.LayerNorm(self.critic_rnn_units)
                    critic_out_size = self.critic_rnn_units
                else:
                    # feedforward critic (unchanged behavior)
                    self.critic_mlp_net = self._build_mlp(
                        input_size=mlp_input_size, units=self.critic_units, **critic_mlp_common
                    )
                    critic_out_size = self.critic_units[-1] if len(self.critic_units) > 0 else mlp_input_size

            # Build Heads for critic (num_tasks, value_size, critic_out_size)
            if self.multi_head:
                self.value_weight = nn.Parameter(torch.zeros(self.num_tasks, self.value_size, critic_out_size))
                self.value_bias = nn.Parameter(torch.zeros(self.num_tasks, self.value_size))
            else:
                self.value = self._build_value_layer(critic_out_size, self.value_size)
            
            self.value_act = self.activations_factory.create(self.value_activation)

            self._build_action_heads(actor_out_size)
            self._init_weights()
            
        def _init_weights(self):
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

            actor_init = self.init_factory.create(**self.actor_initializer)
            critic_init = self.init_factory.create(**self.critic_initializer)
            
            actor_for_init = self.actor_mlp_net if self.has_shared_rnn else self.actor_mlp
            for m in actor_for_init.modules():
                if isinstance(m, nn.Linear):
                    actor_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
            
            if self.separate:
                for m in self.critic_mlp_net.modules():
                    if isinstance(m, nn.Linear):
                        critic_init(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)

            # Initialize value head with critic initializer
            if self.multi_head:
                # Loop to initialize each head independently
                for i in range(self.num_tasks):
                    critic_init(self.value_weight[i])
                    torch.nn.init.zeros_(self.value_bias[i])
            else:
                critic_init(self.value.weight)
                if getattr(self.value, "bias", None) is not None:
                    torch.nn.init.zeros_(self.value.bias)

            # Initialize action heads with actor initializer
            if self.is_discrete:
                if self.multi_head:
                     raise NotImplementedError
                else:
                    actor_init(self.logits.weight)
                    if getattr(self.logits, "bias", None) is not None:
                        torch.nn.init.zeros_(self.logits.bias)
            
            if self.is_multi_discrete:
                if self.multi_head:
                     raise NotImplementedError
                else:
                    for logit_layer in self.logits:
                        actor_init(logit_layer.weight)
                        if getattr(logit_layer, "bias", None) is not None:
                            torch.nn.init.zeros_(logit_layer.bias)
            
            if self.is_continuous:
                if self.multi_head:
                    for i in range(self.num_tasks):
                        mu_init(self.mu_weight[i])
                        torch.nn.init.zeros_(self.mu_bias[i])
                        
                        if not self.fixed_sigma:
                            sigma_init(self.sigma_weight[i])
                            torch.nn.init.zeros_(self.sigma_bias[i])
                    
                    if self.fixed_sigma:
                        sigma_init(self.sigma)
                else:
                    mu_init(self.mu.weight)
                    if getattr(self.mu, "bias", None) is not None:
                        torch.nn.init.zeros_(self.mu.bias)
                    if self.fixed_sigma:
                        sigma_init(self.sigma)
                    else:
                        sigma_init(self.sigma.weight)
                        if getattr(self.sigma, "bias", None) is not None:
                            torch.nn.init.zeros_(self.sigma.bias)
        
        def _build_action_heads(self, input_dim):
            if self.is_discrete:
                if self.multi_head:
                    self.logits_weight = nn.Parameter(torch.zeros(self.num_tasks, self.actions_num, input_dim))
                    self.logits_bias = nn.Parameter(torch.zeros(self.num_tasks, self.actions_num))
                else:
                    self.logits = torch.nn.Linear(input_dim, self.actions_num)
            
            if self.is_multi_discrete:
                if self.multi_head:
                    raise NotImplementedError
                else:
                    self.logits = torch.nn.ModuleList([torch.nn.Linear(input_dim, num) for num in self.actions_num])
            
            if self.is_continuous:
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 

                if self.multi_head:
                    if self.fixed_sigma == 'coef_cond':
                        raise NotImplementedError("fixed_sigma='coef_cond' is not supported together with multi_head=True")
                    # instead of looping through weight for each task in forward, do batched matrix multiply
                    # (Num_Tasks, action_dim, actor_units[-1])
                    self.mu_weight = nn.Parameter(torch.zeros(self.num_tasks, self.actions_num, input_dim))
                    self.mu_bias = nn.Parameter(torch.zeros(self.num_tasks, self.actions_num))

                    # Sigma
                    if self.fixed_sigma:
                        self.sigma = nn.Parameter(torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                    else:
                        self.sigma_weight = nn.Parameter(torch.zeros(self.num_tasks, self.actions_num, input_dim))
                        self.sigma_bias = nn.Parameter(torch.zeros(self.num_tasks, self.actions_num))
                else:
                    self.mu = torch.nn.Linear(input_dim, self.actions_num)
                    if self.fixed_sigma == 'coef_cond':
                        # SAPG per-policy fixed sigma (paper's "each block has its own learnable
                        # vector sigma"). One learnable sigma per policy, selected at forward time
                        # by the policy one-hot. Matches official rl_games sapg coef_cond mode.
                        assert self.num_policies > 0, "fixed_sigma='coef_cond' requires num_policies > 0"
                        self.sigma = nn.Parameter(torch.zeros(self.num_policies, self.actions_num, dtype=torch.float32), requires_grad=True)
                    elif self.fixed_sigma:
                        self.sigma = nn.Parameter(torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                    else:
                        self.sigma = torch.nn.Linear(input_dim, self.actions_num)
    
        def _build_actor(self, actor_args, task_embedding_args, policy_embedding_args):
            actor_mlp = self._build_mlp(**actor_args)
            return Actor(actor_mlp, task_embedding_args, policy_embedding_args)

        def _build_critic(self, critic_args, task_embedding_args, policy_embedding_args):
            critic_mlp = self._build_mlp(**critic_args)
            return Critic(critic_mlp, task_embedding_args, policy_embedding_args)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            task_indices = obs_dict.get('task_indices', None)
            rnn_states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)
            states = None

            if self.has_shared_rnn:
                # Shared recurrent trunk (separate=False): rl_games sequence-reshape +
                # done-masked RNN, output feeds both the action heads and the value head.
                seq_length = obs_dict.get('seq_length', 1)
                a_out = self.actor_task_embedder(self.actor_policy_embedder(obs))
                if not self.critic_rnn_before_mlp:
                    a_out = self.actor_mlp_net(a_out)
                batch_size = a_out.size(0)
                num_seqs = batch_size // seq_length
                a_out = a_out.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                a_dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1) if dones is not None else None
                a_out, a_states = self.a_rnn(a_out, rnn_states, a_dones, bptt_len)
                a_out = a_out.transpose(0, 1).contiguous().reshape(batch_size, -1)
                if self.critic_rnn_ln:
                    a_out = self.a_layer_norm(a_out)
                if not isinstance(a_states, tuple):
                    a_states = (a_states,)
                states = a_states
                if self.critic_rnn_before_mlp:
                    a_out = self.actor_mlp_net(a_out)
            else:
                a_out = self.actor_mlp(obs)

            if self.separate:
                c_out = self.critic_task_embedder(self.critic_policy_embedder(obs))
                if self.has_critic_rnn:
                    # rl_games sequence-reshape + done-masked RNN, critic only
                    seq_length = obs_dict.get('seq_length', 1)
                    if not self.critic_rnn_before_mlp:
                        c_out = self.critic_mlp_net(c_out)
                    batch_size = c_out.size(0)
                    num_seqs = batch_size // seq_length
                    c_out = c_out.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                    c_states = rnn_states
                    c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)
                    c_out = c_out.transpose(0, 1).contiguous().reshape(batch_size, -1)
                    if self.critic_rnn_ln:
                        c_out = self.c_layer_norm(c_out)
                    if not isinstance(c_states, tuple):
                        c_states = (c_states,)
                    states = c_states
                    if self.critic_rnn_before_mlp:
                        c_out = self.critic_mlp_net(c_out)
                else:
                    c_out = self.critic_mlp_net(c_out)
            else:
                c_out = a_out

            # Heads
            if self.multi_head:
                # (B, Val_Size, critic_units[-1])
                v_w = self.value_weight[task_indices] 
                v_b = self.value_bias[task_indices]
                
                # (B, Val_Size, critic_units[-1]) x (B, critic_units[-1], 1) -> (B, Val, 1)
                value = torch.bmm(v_w, c_out.unsqueeze(-1)).squeeze(-1) + v_b
                value = self.value_act(value)
            else:
                value = self.value_act(self.value(c_out))

            if self.central_value:
                return value, states

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits, value, states
            
            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits, value, states
            
            if self.is_continuous:
                if self.multi_head:
                    # (B, Action Dim, actor_units[-1]) i.e we have a weight matrix for each task in the batch
                    mu_w = self.mu_weight[task_indices]
                    mu_b = self.mu_bias[task_indices]
                    
                    # BMM: (B, Action, actor_units[-1]) x (B, actor_units[-1], 1) -> (B, Action, 1)
                    mu = torch.bmm(mu_w, a_out.unsqueeze(-1)).squeeze(-1) + mu_b
                    mu = self.mu_act(mu)

                    if self.fixed_sigma:
                        sigma = self.sigma_act(self.sigma)
                        sigma = sigma.expand_as(mu) # Expand to batch size
                    else:
                        sig_w = self.sigma_weight[task_indices]
                        sig_b = self.sigma_bias[task_indices]
                        sigma = torch.bmm(sig_w, a_out.unsqueeze(-1)).squeeze(-1) + sig_b
                        sigma = self.sigma_act(sigma)
                else:
                    mu = self.mu_act(self.mu(a_out))
                    if self.fixed_sigma == 'coef_cond':
                        # Look up per-policy sigma from raw obs's policy one-hot (still at
                        # the end of obs at this point — PolicyEmbedder runs inside actor_mlp
                        # but `obs` here is the raw obs_dict['obs']).
                        policy_ids = obs[..., -self.num_policies:].argmax(dim=-1)
                        sigma = self.sigma_act(self.sigma[policy_ids])
                    elif self.fixed_sigma:
                        sigma = self.sigma_act(self.sigma)
                        sigma = sigma.expand_as(mu)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma, value, states

        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            num_layers = self.critic_rnn_layers
            units = self.critic_rnn_units
            if self.critic_rnn_name == 'lstm':
                return (
                    torch.zeros((num_layers, self.num_seqs, units)),
                    torch.zeros((num_layers, self.num_seqs, units)),
                )
            # gru / identity: single hidden state
            return (torch.zeros((num_layers, self.num_seqs, units)),)

    def build(self, name, **kwargs):
        net = self.Network(self.params, **kwargs)
        return net
    
class TaskEmbedder(nn.Module):
    def __init__(self, task_embedding_args):
        super().__init__()
        self.learn_embedding = task_embedding_args['learn_task_embedding']
        self.num_tasks = task_embedding_args['num_tasks']
        
        if self.learn_embedding:
            self.embedding = nn.Embedding(
                self.num_tasks, 
                task_embedding_args['task_embedding_dim']
            )
        else:
            self.embedding = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            task_ids_one_hot = obs[..., -self.num_tasks:]
            task_indices = torch.argmax(task_ids_one_hot, dim=1)
            task_embeds = self.embedding(task_indices)
            return torch.cat([obs[..., :-self.num_tasks], task_embeds], dim=-1)
        
        # If not learning embeddings, return obs exactly as is
        return obs

class PolicyEmbedder(nn.Module):
    """SAPG φ_j: learned per-policy parameter that replaces the policy one-hot in obs.
    Mirrors rl_games sapg net_type='extra_param'. φ_j is inserted before the task
    one-hot so TaskEmbedder (which reads from the end) still works.
    """
    def __init__(self, num_policies, embedding_dim, num_tasks, learn):
        super().__init__()
        self.num_policies = num_policies
        self.num_tasks = num_tasks
        self.learn = learn and num_policies > 1
        if self.learn:
            self.phi = nn.Parameter(
                torch.randn(num_policies, embedding_dim, dtype=torch.float32),
                requires_grad=True,
            )
        else:
            self.phi = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.learn:
            return obs
        policy_oh = obs[..., -self.num_policies:]
        rest = obs[..., :-self.num_policies]
        task_oh = rest[..., -self.num_tasks:]
        real = rest[..., :-self.num_tasks]
        phi = self.phi[policy_oh.argmax(dim=-1)]
        return torch.cat([real, phi, task_oh], dim=-1)


class Actor(nn.Module):
    def __init__(self, actor_mlp, task_embedding_args, policy_embedding_args):
        super().__init__()
        self.policy_embedder = PolicyEmbedder(**policy_embedding_args)
        self.embedder = TaskEmbedder(task_embedding_args)
        self.actor_mlp = actor_mlp

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.policy_embedder(obs)
        x = self.embedder(x)
        return self.actor_mlp(x)

class Critic(nn.Module):
    def __init__(self, critic_mlp, task_embedding_args, policy_embedding_args):
        super().__init__()
        self.policy_embedder = PolicyEmbedder(**policy_embedding_args)
        self.embedder = TaskEmbedder(task_embedding_args)
        self.critic_mlp = critic_mlp

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.policy_embedder(obs)
        x = self.embedder(x)
        return self.critic_mlp(x)