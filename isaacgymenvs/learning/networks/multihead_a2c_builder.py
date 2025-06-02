from torch import nn
import torch

from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder
from rl_games.algos_torch import torch_ext


class MultiHeadA2CBuilder(A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):

        def parse_params(self, params):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_rnn = 'rnn' in params
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)
            self.multi_head = params.get('multi_head', False)

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
            self.task_indices = kwargs.pop('task_indices', [0])
            self.num_tasks = torch.unique(self.task_indices).size(0)
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
                    'norm_func_name' : self.normalization,
                }
                self.actor_cnn = self._build_conv(**cnn_args)

                if self.separate:
                    self.critic_cnn = self._build_conv( **cnn_args)

            cnn_output_size = self._calc_input_size(input_shape, self.actor_cnn)

            mlp_input_size = cnn_output_size
            if len(self.units) == 0:
                out_size = cnn_output_size
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    if self.rnn_concat_input:
                        rnn_in_size += cnn_output_size

                    out_size = self.rnn_units
                    if self.rnn_concat_output:
                        out_size += cnn_output_size
                else:
                    rnn_in_size = cnn_output_size

                    mlp_input_size = self.rnn_units
                    if self.rnn_concat_output:
                        mlp_input_size += cnn_output_size

                if self.separate:
                    self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                        self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size' : mlp_input_size,
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            # this is the value network head
            if self.multi_head:
                self.value = torch.nn.ModuleList([self._build_value_layer(out_size, self.value_size) for _ in range(self.num_tasks)])
            else:
                self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                if self.multi_head:
                    raise NotImplementedError
                else:
                    self.logits = torch.nn.Linear(out_size, self.actions_num)
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                if self.multi_head:
                    raise NotImplementedError
                else:
                    self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in self.actions_num])
            if self.is_continuous:
                if self.multi_head:
                    self.mu = torch.nn.ModuleList([torch.nn.Linear(out_size, self.actions_num) for _ in range(self.num_tasks)])
                    self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                    mu_init = self.init_factory.create(**self.space_config['mu_init'])
                    self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                    if self.fixed_sigma:
                        self.sigma = nn.Parameter(torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                    else:
                        self.sigma = torch.nn.ModuleList([torch.nn.Linear(out_size, self.actions_num) for _ in range(self.num_tasks)])
                else:
                    self.mu = torch.nn.Linear(out_size, self.actions_num)
                    self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                    mu_init = self.init_factory.create(**self.space_config['mu_init'])
                    self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                    if self.fixed_sigma:
                        self.sigma = nn.Parameter(torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                    else:
                        self.sigma = torch.nn.Linear(out_size, self.actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            if self.is_continuous:
                if self.multi_head:
                    for l in self.mu:
                        mu_init(l.weight)
                else:
                    mu_init(self.mu.weight)

                if self.multi_head:
                    if self.fixed_sigma:
                        sigma_init(self.sigma)
                    else:
                        for l in self.sigma:
                            sigma_init(l.weight)
                else:
                    if self.fixed_sigma:
                        sigma_init(self.sigma)
                    else:
                        sigma_init(self.sigma.weight)  

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            task_indices = obs_dict.get('task_indices', None)
            unique_task_indices = torch.unique(task_indices)
            bptt_len = obs_dict.get('bptt_len', 0)

            if self.has_cnn:
                # for obs shape 4
                # input expected shape (B, W, H, C)
                # convert to (B, C, W, H)
                if self.permute_input and len(obs.shape) == 4:
                    obs = obs.permute((0, 3, 1, 2))

            if self.separate:
                a_out = c_out = obs
                a_out = self.actor_cnn(a_out)
                a_out = a_out.contiguous().view(a_out.size(0), -1)

                c_out = self.critic_cnn(c_out)
                c_out = c_out.contiguous().view(c_out.size(0), -1)                    

                if self.has_rnn:
                    seq_length = obs_dict.get('seq_length', 1)

                    a_cnn_out = a_out
                    c_cnn_out = c_out
                    if not self.is_rnn_before_mlp:
                        a_out = self.actor_mlp(a_cnn_out)
                        c_out = self.critic_mlp(c_cnn_out)

                        if self.rnn_concat_input:
                            a_out = torch.cat([a_out, a_cnn_out], dim=1)
                            c_out = torch.cat([c_out, c_cnn_out], dim=1)

                    batch_size = a_out.size()[0]
                    num_seqs = batch_size // seq_length
                    a_out = a_out.reshape(num_seqs, seq_length, -1)
                    c_out = c_out.reshape(num_seqs, seq_length, -1)

                    a_out = a_out.transpose(0,1)
                    c_out = c_out.transpose(0,1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0,1)

                    if len(states) == 2:
                        a_states = states[0]
                        c_states = states[1]
                    else:
                        a_states = states[:2]
                        c_states = states[2:]                        
                    a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
                    c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)

                    a_out = a_out.transpose(0,1)
                    c_out = c_out.transpose(0,1)
                    a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
                    c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                    if self.rnn_ln:
                        a_out = self.a_layer_norm(a_out)
                        c_out = self.c_layer_norm(c_out)

                    if type(a_states) is not tuple:
                        a_states = (a_states,)
                        c_states = (c_states,)
                    states = a_states + c_states

                    if self.rnn_concat_output:
                        a_out = torch.cat([a_out, a_cnn_out], dim=1)
                        c_out = torch.cat([c_out, c_cnn_out], dim=1)

                    if self.is_rnn_before_mlp:
                        a_out = self.actor_mlp(a_out)
                        c_out = self.critic_mlp(c_out)
                else:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)
                            
                if self.multi_head:
                    value_all = torch.zeros((obs.shape[0], self.value_size), dtype=torch.float32, device=obs.device)
                    for i, task_index in enumerate(unique_task_indices):
                        mask = task_indices == task_index
                        value = self.value_act(self.value[task_index](c_out[mask]))
                        value_all[mask] = value
                    value = value_all
                else:
                    value = self.value_act(self.value(c_out))

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, value, states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, value, states

                if self.is_continuous:
                    if self.multi_head:
                        mu_all = torch.zeros((obs.shape[0], self.actions_num), dtype=torch.float32, device=obs.device)
                        sigma_all = torch.zeros((obs.shape[0], self.actions_num), dtype=torch.float32, device=obs.device)
                        for i, task_index in enumerate(unique_task_indices):
                            mask = task_indices == task_index
                            mu = self.mu_act(self.mu[task_index](a_out[mask]))
                            if self.fixed_sigma:
                                sigma = mu * 0.0 + self.sigma_act(self.sigma)
                            else:
                                sigma = self.sigma_act(self.sigma[task_index](a_out[mask]))
                            mu_all[mask] = mu
                            sigma_all[mask] = sigma
                        mu = mu_all
                        sigma = sigma_all
                    else:
                        mu = self.mu_act(self.mu(a_out))
                        if self.fixed_sigma:
                            sigma = mu * 0.0 + self.sigma_act(self.sigma)
                        else:
                            sigma = self.sigma_act(self.sigma(a_out))

                    return mu, sigma, value, states
            else:
                out = obs
                out = self.actor_cnn(out)
                out = out.flatten(1)                

                if self.has_rnn:
                    seq_length = obs_dict.get('seq_length', 1)

                    cnn_out = out
                    if not self.is_rnn_before_mlp:
                        out = self.actor_mlp(out)
                        if self.rnn_concat_input:
                            out = torch.cat([out, cnn_out], dim=1)

                    batch_size = out.size()[0]
                    num_seqs = batch_size // seq_length
                    out = out.reshape(num_seqs, seq_length, -1)

                    if len(states) == 1:
                        states = states[0]

                    out = out.transpose(0, 1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0, 1)
                    out, states = self.rnn(out, states, dones, bptt_len)
                    out = out.transpose(0, 1)
                    out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                    if self.rnn_ln:
                        out = self.layer_norm(out)
                    if self.rnn_concat_output:
                        out = torch.cat([out, cnn_out], dim=1)
                    if self.is_rnn_before_mlp:
                        out = self.actor_mlp(out)
                    if type(states) is not tuple:
                        states = (states,)
                else:
                    out = self.actor_mlp(out)
                
                if self.multi_head:
                    value_all = torch.zeros((obs.shape[0], self.value_size), dtype=torch.float32, device=obs.device)
                    for i, task_index in enumerate(unique_task_indices):
                        mask = task_indices == task_index
                        value = self.value_act(self.value[task_index](out[mask]))
                        value_all[mask] = value
                    value = value_all
                else:
                    value = self.value_act(self.value(out))

                if self.central_value:
                    return value, states

                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value, states
                if self.is_multi_discrete:
                    logits = [logit(out) for logit in self.logits]
                    return logits, value, states
                if self.is_continuous:
                    if self.multi_head:
                        mu_all = torch.zeros((obs.shape[0], self.actions_num), dtype=torch.float32, device=obs.device)
                        sigma_all = torch.zeros((obs.shape[0], self.actions_num), dtype=torch.float32, device=obs.device)
                        for i, task_index in enumerate(unique_task_indices):
                            mask = task_indices == task_index
                            mu = self.mu_act(self.mu[task_index](out[mask]))
                            if self.fixed_sigma:
                                sigma = mu * 0.0 + self.sigma_act(self.sigma)
                            else:
                                sigma = self.sigma_act(self.sigma[task_index](out[mask]))
                            mu_all[mask] = mu
                            sigma_all[mask] = sigma
                        mu = mu_all
                        sigma = sigma_all
                    else:
                        mu = self.mu_act(self.mu(out))
                        if self.fixed_sigma:
                            sigma = self.sigma_act(self.sigma)
                        else:
                            sigma = self.sigma_act(self.sigma(out))
                    return mu, mu*0 + sigma, value, states
                    
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
            else:
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)                


    def build(self, name, **kwargs):
        net = self.Network(self.params, **kwargs)
        return net