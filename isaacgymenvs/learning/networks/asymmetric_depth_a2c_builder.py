from torch import nn
import torch

from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder


class AsymmetricDepthA2CBuilder(A2CBuilder):
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
            self.critic_width = params['critic']['critic_width']
            self.critic_blocks = params['critic']['critic_blocks']
            self.critic_initializer = params['critic']['initializer']
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
            critic_out_size = self.critic_width if self.critic_width > 0 else mlp_input_size

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
                    'width' : self.critic_width, 
                    'blocks' : self.critic_blocks,
                }

                self.critic_mlp = self._build_critic(critic_mlp_args, task_embedding_args)

            # Build Heads
            self.value = self._build_value_layer(critic_out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            self._build_action_heads(actor_out_size)
            self._init_weights()
            
        def _init_weights(self):
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

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
        
        def _build_action_heads(self, input_dim):
            if self.is_discrete:
                self.logits = torch.nn.Linear(input_dim, self.actions_num)
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(input_dim, num) for num in self.actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(input_dim, self.actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(input_dim, self.actions_num)

        def _build_actor(self, actor_args, task_embedding_args):
            actor_mlp = self._build_mlp(**actor_args)

            return Actor(actor_mlp, task_embedding_args)

        def _build_critic(self, critic_args, task_embedding_args):
            return Critic(critic_args, task_embedding_args)

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

class TaskEmbedder(nn.Module):
    """Handles task embedding logic"""
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

    def forward(self, obs):
        if self.embedding is not None:
            # Extract one-hot part from the end of obs
            task_ids_one_hot = obs[..., -self.num_tasks:]
            task_indices = torch.argmax(task_ids_one_hot, dim=1)
            task_embeds = self.embedding(task_indices)
            
            # Concat the base observation with the learned embedding
            return torch.cat([obs[..., :-self.num_tasks], task_embeds], dim=-1)
        return obs

class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        layers = []
        for _ in range(4):
            layers.append(nn.Linear(width, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # The Residual Connection: f(x) + x
        return self.net(x) + x

class Actor(nn.Module):
    def __init__(self, actor_mlp, task_embedding_args):
        super().__init__()
        self.actor_mlp = actor_mlp

        self.embedder = TaskEmbedder(task_embedding_args)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.embedder(obs)
        return self.actor_mlp(x)

class Critic(nn.Module):
    def __init__(self, critic_args, task_embedding_args):
        super().__init__()
        self.embedder = TaskEmbedder(task_embedding_args)

        input_size = critic_args['input_size']
        width = critic_args['width']
        num_blocks = critic_args['blocks']

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, width),
            nn.LayerNorm(width),
            nn.SiLU()
        )
        
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(num_blocks)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.embedder(obs)
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        return x