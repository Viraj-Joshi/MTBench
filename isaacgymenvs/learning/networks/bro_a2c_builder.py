import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from rl_games.algos_torch import network_builder

from typing import List, Tuple
import math

class BROA2CBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = BROA2CBuilder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            action_dim = kwargs.pop('actions_num')
            obs_shape = kwargs.pop('input_shape')
            num_envs = kwargs['task_indices'].shape[0]
            device = kwargs.pop('device')
            learn_task_embedding = kwargs.pop('learn_task_embedding', False)
            num_tasks = torch.unique(kwargs['task_indices']).shape[0]
            task_embedding_dim = kwargs.pop('task_embedding_dim')
            # dim of obs without one hot task embedding
            real_obs_dim = obs_shape[0] - num_tasks

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            self.separate = params.get('separate', False)
            
            if not self.separate:
                raise NotImplementedError("BRO + PPO only supports separate actor and critic networks for now.")

            # obs dim changes to real_obs_dim + task_embedding_dim if we are using learnable task embeddings
            obs_dim = real_obs_dim + task_embedding_dim if learn_task_embedding else obs_shape[0]
            actor_args = {
                'n_obs' : obs_dim,
                'n_act' : action_dim,
                'num_envs' : num_envs,
                'hidden_dim' : self.actor_hidden_dim,
                'num_blocks' : self.actor_num_blocks,
                'learn_task_embedding' : learn_task_embedding,
                'num_tasks': num_tasks,
                'task_embedding_dim' : task_embedding_dim,
                'device' : device,
            }

            critic_args = {
                'n_obs' : obs_dim,
                'n_act' : action_dim,
                'dv': self.dv,
                'hidden_dim' : self.critic_hidden_dim,
                'num_blocks' : self.critic_num_blocks,
                'learn_task_embedding' : learn_task_embedding,
                'num_tasks': num_tasks,
                'task_embedding_dim' : task_embedding_dim,
                'device' : device,
            }
            
            print("Building Actor")
            self.actor = self._build_actor(actor_args)

            print("Building Critic")
            self.critic = self._build_critic(critic_args)

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

        def _build_critic(self, critic_args):
            n_obs = critic_args['n_obs']
            n_act = critic_args['n_act']
            critic_hidden_dim = critic_args['hidden_dim']
            critic_num_blocks = critic_args['num_blocks']
            device = critic_args['device']
            learn_task_embedding = critic_args['learn_task_embedding']
            num_tasks = critic_args['num_tasks']
            task_embedding_dim = critic_args['task_embedding_dim']
            # optional distributional value function 
            dv = critic_args['dv'] 

            return Critic(
                n_obs=n_obs,
                n_act=n_act,
                dv=dv,
                hidden_dim=critic_hidden_dim,
                num_blocks=critic_num_blocks,
                learn_task_embedding=learn_task_embedding,
                num_tasks=num_tasks,
                task_embedding_dim=task_embedding_dim,
                device=device,
            )

        def _build_actor(self,actor_args):
            n_obs = actor_args['n_obs']
            n_act = actor_args['n_act']
            num_envs = actor_args['num_envs']
            actor_hidden_dim = actor_args['hidden_dim']
            actor_num_blocks = actor_args['num_blocks']
            device = actor_args['device']
            learn_task_embedding = actor_args['learn_task_embedding']
            num_tasks = actor_args['num_tasks']
            task_embedding_dim = actor_args['task_embedding_dim']

            return Actor(
                n_obs=n_obs,
                n_act=n_act,
                num_envs=num_envs,
                hidden_dim=actor_hidden_dim,
                num_blocks=actor_num_blocks,
                learn_task_embedding=learn_task_embedding,
                num_tasks=num_tasks,
                task_embedding_dim=task_embedding_dim,
                device=device,
            )
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
 
        def is_separate_critic(self):
            return self.separate

        def load(self, params):
            self.actor_hidden_dim = params['actor']['hidden_dim']
            self.actor_num_blocks = params['actor']['num_blocks']

            self.actor_scaler_init = math.sqrt(2.0 / self.actor_hidden_dim)
            self.actor_scaler_scale = math.sqrt(2.0 / self.actor_hidden_dim)
            self.actor_alpha_init = 1.0 / (self.actor_num_blocks + 1)
            self.actor_alpha_scale = 1.0 / math.sqrt(self.actor_hidden_dim)
            self.actor_expansion = 4
            self.actor_c_shift = 3.0

            self.critic_hidden_dim = params['critic']['hidden_dim']
            self.critic_num_blocks = params['critic']['num_blocks']
            self.dv = None
            if 'dv' in params['critic']:  
                self.dv = {
                    'v_min': params['critic']['dv']['v_min'],
                    'v_max': params['critic']['dv']['v_max'],
                    'num_atoms': params['critic']['dv']['num_atoms']
                }              

            self.critic_scaler_init = math.sqrt(2.0 / self.critic_hidden_dim)
            self.critic_scaler_scale = math.sqrt(2.0 / self.critic_hidden_dim)
            self.critic_alpha_init = 1.0 / (self.critic_num_blocks + 1)
            self.critic_alpha_scale = 1.0 / math.sqrt(self.critic_hidden_dim)
            self.critic_expansion = 4
            self.critic_c_shift = 3.0

            self.has_space = 'space' in params
            self.value_shape = params.get('value_shape', 1)
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

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

def layer_init(layer):
    # torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
    return layer

class Critic(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        num_blocks: int,
        learn_task_embedding: bool,
        num_tasks: int,
        task_embedding_dim: int,
        dv: dict,
        device: torch.device = None,
    ):
        super().__init__()
        self.is_distributional = dv is not None

        if self.is_distributional:
            v_min = dv['v_min']
            v_max = dv['v_max']
            num_atoms = dv['num_atoms']
            self.critic = None

            self.register_buffer(
            "q_support", torch.linspace(v_min, v_max, num_atoms, device=device)
            )
            self.device = device
        else:
            embedder = nn.Sequential(
                layer_init(nn.Linear(n_obs, hidden_dim)),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )

            encoder = nn.Sequential(
                *[
                    BRONetBlock(
                        in_dim=hidden_dim,
                        hidden_dim=hidden_dim
                    )
                    for _ in range(num_blocks)
                ]
            )
            predictor = layer_init(nn.Linear(hidden_dim, 1))

            self.critic = nn.Sequential(
                embedder,
                encoder,
                predictor,
            )
        
        self.task_embedding = torch.nn.Embedding(
            num_embeddings=num_tasks,
            embedding_dim=task_embedding_dim,
            max_norm=1.0,
        ) if learn_task_embedding else None

        self.num_tasks = num_tasks

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.task_embedding is not None:
            task_ids_one_hot = obs[..., -self.num_tasks :]
            task_indices = torch.argmax(task_ids_one_hot, dim=1)
            task_embeddings = self.task_embedding(task_indices)
            obs = torch.cat([obs[..., : -self.num_tasks], task_embeddings], dim=-1)
        x = self.critic(obs)
        return x

    def projection(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
    ) -> torch.Tensor:
        """Projection operation that includes q_support directly"""
        if not self.is_distributional:
            raise ValueError("Critic is not distributional, cannot project.")
        q1_proj = self.critic.projection(
            obs,
            rewards,
            bootstrap,
            discount,
            self.q_support,
            self.q_support.device,
        )
        
        return q1_proj

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        if not self.is_distributional:
            raise ValueError("Critic is not distributional, cannot get value.")
        return torch.sum(probs * self.q_support, dim=1)


class Actor(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        hidden_dim: int,
        num_blocks: int,
        learn_task_embedding: bool,
        num_tasks: int,
        task_embedding_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_act = n_act

        self.embedder = nn.Sequential(
            layer_init(nn.Linear(n_obs, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.encoder = nn.Sequential(
            *[
                BRONetBlock(
                    in_dim=hidden_dim,
                    hidden_dim=hidden_dim
                )
                for _ in range(num_blocks)
            ]
        )
        self.predictor = layer_init(nn.Linear(hidden_dim, n_act))

        self.task_embedding = torch.nn.Embedding(
            num_embeddings=num_tasks,
            embedding_dim=task_embedding_dim,
            max_norm=1.0,
        ) if learn_task_embedding else None

        self.num_tasks = num_tasks
        self.n_envs = num_envs
        self.device = device

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.task_embedding is not None:
            task_ids_one_hot = obs[..., -self.num_tasks :]
            task_indices = torch.argmax(task_ids_one_hot, dim=1)
            task_embeddings = self.task_embedding(task_indices)
            obs = torch.cat([obs[..., : -self.num_tasks], task_embeddings], dim=-1)
        x = self.embedder(obs)
        x = self.encoder(x)
        x = self.predictor(x)
        return x

class BRONetBlock(nn.Module):
    """
    A residual block following BRO.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.block = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, in_dim)),
            nn.LayerNorm(hidden_dim),
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block.
        """
        out = self.block(x)
        return x + out

