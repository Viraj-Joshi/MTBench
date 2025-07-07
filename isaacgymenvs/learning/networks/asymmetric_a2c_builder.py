from torch import nn
import torch

from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder
from rl_games.algos_torch import torch_ext


class AsymmetricA2CBuilder(A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):

        def parse_params(self, params):
            self.separate = params.get('separate', False)
            
            # Actor parameters
            self.actor_units = params['actor']['units']
            self.actor_activation = params['actor']['activation']
            self.actor_initializer = params['actor']['initializer']
            self.actor_is_d2rl = params['actor'].get('d2rl', False)
            self.actor_norm_only_first_layer = params['actor'].get('norm_only_first_layer', False)
            self.actor_normalization = params.get('normalization', None) # Note: user code had params.get, might want params['actor'].get

            # Critic parameters
            self.critic_units = params['critic']['units']
            self.critic_activation = params['critic']['activation']
            self.critic_initializer = params['critic']['initializer']
            self.critic_is_d2rl = params['critic'].get('d2rl', False)
            self.critic_norm_only_first_layer = params['critic'].get('norm_only_first_layer', False)
            self.critic_normalization = params['critic'].get('normalization', None)
            self.value_activation = params.get('value_activation', 'None')

            self.has_rnn = 'rnn' in params
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

            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.rnn_ln = params['rnn'].get('layer_norm', False)
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
                self.rnn_concat_input = params['rnn'].get('concat_input', False)
                self.rnn_concat_output = params['rnn'].get('concat_output', False)

            if 'cnn' in params:
                self.has_cnn = True
                self.cnn = params['cnn']
                self.permute_input = self.cnn.get('permute_input', True)
            else:
                self.has_cnn = False
                
        def __init__(self, params, **kwargs):
            self.actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.parse_params(params)
            
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()
            
            if self.has_cnn:
                if self.permute_input:
                    input_shape = torch_ext.shape_whc_to_cwh(input_shape)
                cnn_args = {
                    'ctype' : self.cnn['type'], 
                    'input_shape' : input_shape, 
                    'convs' :self.cnn['convs'], 
                    'activation' : self.cnn['activation'], 
                    'norm_func_name' : self.actor_normalization,
                }
                self.actor_cnn = self._build_conv(**cnn_args)

                if self.separate:
                    # use critic normalization for critic cnn
                    cnn_args['norm_func_name'] = self.critic_normalization
                    self.critic_cnn = self._build_conv(**cnn_args)

            cnn_output_size = self._calc_input_size(input_shape, self.actor_cnn)

            # Core logic for calculating network sizes
            mlp_input_size = cnn_output_size
            
            # Final feature size for actor and critic heads
            actor_out_size = cnn_output_size
            critic_out_size = cnn_output_size

            if self.has_rnn:
                if self.is_rnn_before_mlp:
                    # Path: CNN -> RNN -> MLP
                    rnn_in_size = cnn_output_size
                    mlp_input_size = self.rnn_units
                    if self.rnn_concat_output:
                        mlp_input_size += cnn_output_size
                    
                    actor_out_size = self.actor_units[-1] if len(self.actor_units) > 0 else mlp_input_size
                    critic_out_size = self.critic_units[-1] if len(self.critic_units) > 0 else mlp_input_size
                else:
                    # Path: CNN -> MLP -> RNN
                    a_rnn_in_size = self.actor_units[-1] if len(self.actor_units) > 0 else cnn_output_size
                    c_rnn_in_size = self.critic_units[-1] if len(self.critic_units) > 0 else cnn_output_size
                    
                    if self.rnn_concat_input:
                        a_rnn_in_size += cnn_output_size
                        c_rnn_in_size += cnn_output_size

                    actor_out_size = self.rnn_units
                    critic_out_size = self.rnn_units
                    if self.rnn_concat_output:
                        actor_out_size += cnn_output_size
                        critic_out_size += cnn_output_size
                
                # Build RNNs
                if self.separate:
                    self.a_rnn = self._build_rnn(self.rnn_name, a_rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.c_rnn = self._build_rnn(self.rnn_name, c_rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                        self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    # If not separate, actor and critic RNNs are the same
                    rnn_in_size = a_rnn_in_size if not self.is_rnn_before_mlp else cnn_output_size
                    self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.layer_norm = torch.nn.LayerNorm(self.rnn_units)
            else: # No RNN
                # Path: CNN -> MLP
                actor_out_size = self.actor_units[-1] if len(self.actor_units) > 0 else cnn_output_size
                critic_out_size = self.critic_units[-1] if len(self.critic_units) > 0 else cnn_output_size

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
                'norm_only_first_layer' : self.actor_norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**actor_mlp_args)
            
            if self.separate:
                critic_mlp_args = {
                    'input_size' : mlp_input_size, 
                    'units' : self.critic_units, 
                    'activation' : self.critic_activation,
                    'norm_func_name' : self.critic_normalization, 
                    'dense_func' : torch.nn.Linear,
                    'd2rl' : self.critic_is_d2rl, 
                    'norm_only_first_layer' : self.critic_norm_only_first_layer
                }
                self.critic_mlp = self._build_mlp(**critic_mlp_args)

            # Build Heads
            self.value = self._build_value_layer(critic_out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(actor_out_size, self.actions_num)
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(actor_out_size, num) for num in self.actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(actor_out_size, self.actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(actor_out_size, self.actions_num)

            # Initialization
            actor_init = self.init_factory.create(**self.actor_initializer)
            critic_init = self.init_factory.create(**self.critic_initializer)
            
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])
                for m_list in [self.actor_cnn, self.critic_cnn]:
                     for m in m_list.modules():
                        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                            cnn_init(m.weight)
                            if getattr(m, "bias", None) is not None:
                                torch.nn.init.zeros_(m.bias)
            
            for m in self.actor_mlp.modules():
                if isinstance(m, nn.Linear):
                    actor_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
            
            if self.separate:
                for m in self.critic_mlp.modules():
                    if isinstance(m, nn.Linear):
                        critic_init(m.weight)
                        if getattr(m, "bias", None) is not None:
                            torch.nn.init.zeros_(m.bias)

            # Initialize value head with critic initializer
            critic_init(self.value.weight)
            if getattr(self.value, "bias", None) is not None:
                torch.nn.init.zeros_(self.value.bias)

            # Initialize action heads with actor initializer
            if self.is_discrete:
                actor_init(self.logits.weight)
                if getattr(self.logits, "bias", None) is not None:
                    torch.nn.init.zeros_(self.logits.bias)
            if self.is_multi_discrete:
                for logit_layer in self.logits:
                    actor_init(logit_layer.weight)
                    if getattr(logit_layer, "bias", None) is not None:
                        torch.nn.init.zeros_(logit_layer.bias)
            if self.is_continuous:
                mu_init(self.mu.weight)
                if getattr(self.mu, "bias", None) is not None:
                    torch.nn.init.zeros_(self.mu.bias)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)
                    if getattr(self.sigma, "bias", None) is not None:
                        torch.nn.init.zeros_(self.sigma.bias)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)

            if self.has_cnn:
                if self.permute_input and len(obs.shape) == 4:
                    obs = obs.permute((0, 3, 1, 2))

            # Critic Path
            if self.separate:
                c_out = self.critic_cnn(obs)
                c_out = c_out.contiguous().view(c_out.size(0), -1)
            
                if self.has_rnn and self.is_rnn_before_mlp:
                    c_rnn_input = c_out
                    # Critic RNN forward
                    batch_size = c_rnn_input.size()[0]
                    seq_length = obs_dict.get('seq_length', 1)
                    num_seqs = batch_size // seq_length
                    c_rnn_input = c_rnn_input.reshape(num_seqs, seq_length, -1)
                    c_rnn_input = c_rnn_input.transpose(0,1)

                    c_states = states[2:] if self.rnn_name == 'lstm' else states[1]
                    if dones is not None:
                        dones_c = dones.reshape(num_seqs, seq_length, -1).transpose(0,1)
                    else:
                        dones_c = None

                    c_rnn_output, c_states_out = self.c_rnn(c_rnn_input, c_states, dones_c, bptt_len)
                    c_rnn_output = c_rnn_output.transpose(0,1).contiguous().reshape(batch_size, -1)
                    if self.rnn_ln:
                        c_rnn_output = self.c_layer_norm(c_rnn_output)
                    if self.rnn_concat_output:
                        c_out = torch.cat([c_rnn_output, c_out], dim=1)
                    else:
                        c_out = c_rnn_output
                    
                c_out = self.critic_mlp(c_out)

                if self.has_rnn and not self.is_rnn_before_mlp:
                    c_cnn_out = self.critic_cnn(obs).contiguous().view(obs.size(0), -1)
                    if self.rnn_concat_input:
                        c_out = torch.cat([c_out, c_cnn_out], dim=1)
                    # Critic RNN forward
                    batch_size = c_out.size()[0]
                    seq_length = obs_dict.get('seq_length', 1)
                    num_seqs = batch_size // seq_length
                    c_out = c_out.reshape(num_seqs, seq_length, -1).transpose(0,1)

                    c_states = states[2:] if self.rnn_name == 'lstm' else states[1]
                    if dones is not None:
                        dones_c = dones.reshape(num_seqs, seq_length, -1).transpose(0,1)
                    else:
                        dones_c = None

                    c_out, c_states_out = self.c_rnn(c_out, c_states, dones_c, bptt_len)
                    c_out = c_out.transpose(0,1).contiguous().reshape(batch_size, -1)
                    if self.rnn_ln:
                        c_out = self.c_layer_norm(c_out)
                    if self.rnn_concat_output:
                        c_out = torch.cat([c_out, c_cnn_out], dim=1)
            
            # Actor Path
            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            
            if self.has_rnn and self.is_rnn_before_mlp:
                a_rnn_input = a_out
                # Actor RNN forward
                batch_size = a_rnn_input.size()[0]
                seq_length = obs_dict.get('seq_length', 1)
                num_seqs = batch_size // seq_length
                a_rnn_input = a_rnn_input.reshape(num_seqs, seq_length, -1).transpose(0,1)

                a_states = states[:2] if self.rnn_name == 'lstm' else states[0]
                if dones is not None:
                    dones_a = dones.reshape(num_seqs, seq_length, -1).transpose(0,1)
                else:
                    dones_a = None
                
                a_rnn_output, a_states_out = self.a_rnn(a_rnn_input, a_states, dones_a, bptt_len) if self.separate else self.rnn(a_rnn_input, states, dones_a, bptt_len)
                a_rnn_output = a_rnn_output.transpose(0,1).contiguous().reshape(batch_size, -1)
                if self.rnn_ln:
                    a_rnn_output = self.a_layer_norm(a_rnn_output) if self.separate else self.layer_norm(a_rnn_output)
                if self.rnn_concat_output:
                    a_out = torch.cat([a_rnn_output, a_out], dim=1)
                else:
                    a_out = a_rnn_output

            a_out = self.actor_mlp(a_out)
            
            if self.has_rnn and not self.is_rnn_before_mlp:
                a_cnn_out = self.actor_cnn(obs).contiguous().view(obs.size(0), -1)
                if self.rnn_concat_input:
                    a_out = torch.cat([a_out, a_cnn_out], dim=1)
                # Actor RNN forward
                batch_size = a_out.size()[0]
                seq_length = obs_dict.get('seq_length', 1)
                num_seqs = batch_size // seq_length
                a_out = a_out.reshape(num_seqs, seq_length, -1).transpose(0,1)
                
                a_states = states[:2] if self.rnn_name == 'lstm' and self.separate else states
                if dones is not None:
                    dones_a = dones.reshape(num_seqs, seq_length, -1).transpose(0,1)
                else:
                    dones_a = None

                a_out, a_states_out = self.a_rnn(a_out, a_states, dones_a, bptt_len) if self.separate else self.rnn(a_out, states, dones_a, bptt_len)
                a_out = a_out.transpose(0,1).contiguous().reshape(batch_size, -1)
                if self.rnn_ln:
                    a_out = self.a_layer_norm(a_out) if self.separate else self.layer_norm(a_out)
                if self.rnn_concat_output:
                    a_out = torch.cat([a_out, a_cnn_out], dim=1)

            # Combine states
            if self.has_rnn:
                if self.separate:
                    if type(a_states_out) is not tuple: a_states_out = (a_states_out,)
                    if type(c_states_out) is not tuple: c_states_out = (c_states_out,)
                    states = a_states_out + c_states_out
                else:
                    states = a_states_out
            
            # Heads
            value = self.value_act(self.value(c_out if self.separate else a_out))

            if self.central_value:
                return value, states

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits, value, states
            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits, value, states
            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.fixed_sigma:
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
            num_layers = self.rnn_layers
            
            if self.rnn_name == 'identity':
                rnn_units = 1
            else:
                rnn_units = self.rnn_units
            
            if self.rnn_name == 'lstm':
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
            else: # GRU / RNN
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)


    def build(self, name, **kwargs):
        net = self.Network(self.params, **kwargs)
        return net