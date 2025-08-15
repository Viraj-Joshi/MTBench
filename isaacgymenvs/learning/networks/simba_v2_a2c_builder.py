import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from rl_games.algos_torch import network_builder
from isaacgymenvs.learning.networks.simba_v2 import (
    HyperEmbedder,
    HyperLERPBlock,
    HyperPolicy,
    HyperCategoricalValue
)

from typing import List, Tuple
import math

class SimbaV2A2CBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = SimbaV2A2CBuilder.Network(self.params, **kwargs)
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
                raise NotImplementedError("SimbaV2 + PPO only supports separate actor and critic networks for now.")

            # obs dim changes to real_obs_dim + task_embedding_dim if we are using learnable task embeddings
            obs_dim = real_obs_dim + task_embedding_dim if learn_task_embedding else obs_shape[0]
            actor_args = {
                'n_obs' : obs_dim,
                'n_act' : action_dim,
                'num_envs' : num_envs,
                'hidden_dim' : self.actor_hidden_dim,
                'scaler_init' : self.actor_scaler_init,
                'scaler_scale' : self.actor_scaler_scale,
                'alpha_init' : self.actor_alpha_init,
                'alpha_scale' : self.actor_alpha_scale,
                'expansion' : self.actor_expansion,
                'c_shift' : self.actor_c_shift,
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
                'scaler_init' : self.critic_scaler_init,
                'scaler_scale' : self.critic_scaler_scale,
                'alpha_init' : self.critic_alpha_init,
                'alpha_scale' : self.critic_alpha_scale,
                'expansion' : self.critic_expansion,
                'c_shift' : self.critic_c_shift,
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
            # Simba Specific parameters
            critic_scaler_init = critic_args['scaler_init']
            critic_scaler_scale = critic_args['scaler_scale']
            critic_alpha_init = critic_args['alpha_init']
            critic_alpha_scale = critic_args['alpha_scale']
            critic_expansion = critic_args['expansion']
            critic_c_shift = critic_args['c_shift']
            # optional distributional value function 
            dv = critic_args['dv'] 

            return Critic(
                n_obs=n_obs,
                n_act=n_act,
                dv=dv,
                hidden_dim=critic_hidden_dim,
                scaler_init=critic_scaler_init,
                scaler_scale=critic_scaler_scale,
                alpha_init=critic_alpha_init,
                alpha_scale=critic_alpha_scale,
                expansion=critic_expansion,
                c_shift=critic_c_shift,
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
            # Simba Specific parameters
            scaler_init = actor_args['scaler_init']
            scaler_scale = actor_args['scaler_scale']
            alpha_init = actor_args['alpha_init']
            alpha_scale = actor_args['alpha_scale']
            expansion = actor_args['expansion']
            c_shift = actor_args['c_shift']

            return Actor(
                n_obs=n_obs,
                n_act=n_act,
                num_envs=num_envs,
                hidden_dim=actor_hidden_dim,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                alpha_init=alpha_init,
                alpha_scale=alpha_scale,
                expansion=expansion,
                c_shift=c_shift,
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

class DistributionalVNetwork(nn.Module):
    def __init__(
        self,
        n_obs: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        num_blocks: int,
        c_shift: float,
        expansion: int,
        device: torch.device = None,
    ):
        super().__init__()

        self.embedder = HyperEmbedder(
            in_dim=n_obs,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
            device=device,
        )

        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                    expansion=expansion,
                    device=device,
                )
                for _ in range(num_blocks)
            ]
        )

        self.predictor = HyperCategoricalValue(
            hidden_dim=hidden_dim,
            num_bins=num_atoms,
            scaler_init=1.0,
            scaler_scale=1.0,
            device=device,
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        x = self.embedder(x)
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = (
            rewards.unsqueeze(1)
            + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        )
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        l_mask = torch.logical_and((u > 0), (l == u))
        u_mask = torch.logical_and((l < (self.num_atoms - 1)), (l == u))

        l = torch.where(l_mask, l - 1, l)
        u = torch.where(u_mask, u + 1, u)

        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )
        return proj_dist

class Critic(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        dv: dict,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        num_blocks: int,
        c_shift: float,
        expansion: int,
        learn_task_embedding: bool,
        num_tasks: int,
        task_embedding_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.is_distributional = dv is not None

        if self.is_distributional:
            v_min = dv['v_min']
            v_max = dv['v_max']
            num_atoms = dv['num_atoms']
            self.critic = DistributionalVNetwork(
                n_obs=n_obs,
                num_atoms=num_atoms,
                v_min=v_min,
                v_max=v_max,
                hidden_dim=hidden_dim,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                alpha_init=alpha_init,
                alpha_scale=alpha_scale,
                num_blocks=num_blocks,
                c_shift=c_shift,
                expansion=expansion,
                device=device,
            )

            self.register_buffer(
            "q_support", torch.linspace(v_min, v_max, num_atoms, device=device)
            )
            self.device = device
        else:
            self.critic = nn.Sequential(
                HyperEmbedder(
                    in_dim=n_obs,
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    c_shift=c_shift,
                    device=device,
                ),
                *[
                    HyperLERPBlock(
                        hidden_dim=hidden_dim,
                        scaler_init=scaler_init,
                        scaler_scale=scaler_scale,
                        alpha_init=alpha_init,
                        alpha_scale=alpha_scale,
                        expansion=expansion,
                        device=device,
                    )
                    for _ in range(num_blocks)
                ],
                HyperPolicy(
                    hidden_dim=hidden_dim,
                    action_dim=1,
                    scaler_init=1.0,
                    scaler_scale=1.0,
                    device=device,
                )
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
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        expansion: int,
        c_shift: float,
        num_blocks: int,
        learn_task_embedding: bool,
        num_tasks: int,
        task_embedding_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_act = n_act

        self.embedder = HyperEmbedder(
            in_dim=n_obs,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
            device=device,
        )
        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                    expansion=expansion,
                    device=device,
                )
                for _ in range(num_blocks)
            ]
        )
        self.predictor = HyperPolicy(
            hidden_dim=hidden_dim,
            action_dim=n_act,
            scaler_init=1.0,
            scaler_scale=1.0,
            device=device,
        )

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

