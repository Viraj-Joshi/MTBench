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
            num_tasks = torch.unique(kwargs['task_indices']).shape[0]
            task_embedding_dim = kwargs.pop('task_embedding_dim')
            real_obs_dim = input_shape[0] - num_tasks
            obs_dim = real_obs_dim + task_embedding_dim if learn_task_embedding else input_shape[0]
            
            NetworkBuilder.BaseNetwork.__init__(self)
            self.parse_params(params)
            
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
                'num_tasks' : num_tasks
            }
            self.actor_mlp = self._build_actor(actor_mlp_args, task_embedding_args)
            
            if self.separate:
                critic_mlp_args = {
                    'input_size' : mlp_input_size, 
                    'units' : self.critic_units, 
                    'activation' : self.critic_activation,
                    'norm_func_name' : self.critic_normalization, 
                    'dense_func' : torch.nn.Linear,
                    'd2rl' : self.critic_is_d2rl, 
                    'norm_only_first_layer' : self.critic_norm_only_first_layer,
                }

                self.critic_mlp = self._build_critic(critic_mlp_args, task_embedding_args)

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

        def _build_actor(self, actor_args, task_embedding_args):
            actor_mlp = self._build_mlp(**actor_args)

            return Actor(actor_mlp, task_embedding_args)

        def _build_critic(self, critic_args, task_embedding_args):
            critic_mlp = self._build_mlp(**critic_args)

            return Critic(critic_mlp, task_embedding_args)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = None

            # Actor Path
            a_out = self.actor_mlp(obs)

            # Critic Path
            if self.separate:
                critic_obs = obs.clone()

                c_out = self.critic_mlp(obs)
            else:
                c_out = a_out
            
            # Heads
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
                mu = self.mu_act(self.mu(a_out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                    sigma = sigma.expand_as(mu)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))
                return mu, sigma, value, states

        def is_separate_critic(self):
            return self.separate

    def build(self, name, **kwargs):
        net = self.Network(self.params, **kwargs)
        return net

class Actor(nn.Module):
    def __init__(self, actor_mlp, task_embedding_args):
        super().__init__()
        self.actor_mlp = actor_mlp

        learn_task_embedding = task_embedding_args['learn_task_embedding']
        task_embedding_dim = task_embedding_args['task_embedding_dim']
        self.num_tasks = task_embedding_args['num_tasks']

        self.task_embedding = None
        if learn_task_embedding:
            self.task_embedding = nn.Embedding(self.num_tasks, task_embedding_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.task_embedding is not None:
            task_ids_one_hot = obs[..., -self.num_tasks :]
            task_indices = torch.argmax(task_ids_one_hot, dim=1)
            task_embeddings = self.task_embedding(task_indices)
            obs = torch.cat([obs[..., : -self.num_tasks], task_embeddings], dim=-1)
        x = self.actor_mlp(obs)
        return x
    
class Critic(nn.Module):
    def __init__(self, critic_mlp, task_embedding_args):
        super().__init__()
        self.critic_mlp = critic_mlp

        learn_task_embedding = task_embedding_args['learn_task_embedding']
        task_embedding_dim = task_embedding_args['task_embedding_dim']
        self.num_tasks = task_embedding_args['num_tasks']

        self.task_embedding = None
        if learn_task_embedding:
            self.task_embedding = nn.Embedding(self.num_tasks, task_embedding_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.task_embedding is not None:
            task_ids_one_hot = obs[..., -self.num_tasks :]
            task_indices = torch.argmax(task_ids_one_hot, dim=1)
            task_embeddings = self.task_embedding(task_indices)
            obs = torch.cat([obs[..., : -self.num_tasks], task_embeddings], dim=-1)
        x = self.critic_mlp(obs)
        return x