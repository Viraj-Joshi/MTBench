import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from rl_games.algos_torch import network_builder

from typing import List, Tuple

class FastTD3Builder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = FastTD3Builder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            action_shape = kwargs.pop('action_shape')
            obs_shape = kwargs.pop('input_shape')
            num_envs = kwargs.pop('num_envs')
            device = kwargs.pop('device')
            learn_task_embedding = kwargs.pop('learn_task_embedding', False)
            num_tasks = torch.unique(kwargs['task_indices']).shape[0]
            task_embedding_dim = kwargs.pop('task_embedding_dim')
            # get the dim of the real part of the obs
            real_obs_dim = obs_shape[0] - num_tasks

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            # obs dim changes to real_obs_dim + task_embedding_dim if we are using learnable task embeddings
            obs_dim = real_obs_dim + task_embedding_dim if learn_task_embedding else obs_shape[0]
            actor_args = {
                'n_obs' : obs_dim,
                'n_act' : action_shape[0],
                'num_envs' : num_envs,
                'init_scale' : self.init_scale,
                'std_min' : self.std_min,
                'std_max' : self.std_max,
                'hidden_dim' : self.actor_hidden_dim,
                'learn_task_embedding' : learn_task_embedding,
                'task_embedding_dim' : task_embedding_dim,
                'num_tasks' : num_tasks,
                'device' : device,
            }

            critic_args = {
                'n_obs' : obs_dim,
                'n_act' : action_shape[0],
                'num_atoms' : self.num_atoms,
                'v_min' : self.v_min,
                'v_max' : self.v_max,
                'hidden_dim' : self.critic_hidden_dim,
                'learn_task_embedding' : learn_task_embedding,
                'task_embedding_dim' : task_embedding_dim,
                'num_tasks' : num_tasks,
                'device' : device,
            }
            
            print("Building Actor")
            self.actor = self._build_actor(actor_args)

            print("Building Critic")
            self.critic = self._build_critic(critic_args)
            print("Building Critic Target")
            self.critic_target = self._build_critic(critic_args)
            self.critic_target.load_state_dict(self.critic.state_dict())

        def _build_critic(self, critic_args):
            n_obs = critic_args['n_obs']
            n_act = critic_args['n_act']
            num_atoms = critic_args['num_atoms']
            v_min = critic_args['v_min']
            v_max = critic_args['v_max']
            critic_hidden_dim = critic_args['hidden_dim']
            learn_task_embedding = critic_args['learn_task_embedding']
            task_embedding_dim = critic_args['task_embedding_dim']
            num_tasks = critic_args['num_tasks']
            device = critic_args['device']

            return Critic(
                n_obs=n_obs,
                n_act=n_act,
                num_atoms=num_atoms,
                v_min=v_min,
                v_max=v_max,
                hidden_dim=critic_hidden_dim,
                learn_task_embedding=learn_task_embedding,
                task_embedding_dim=task_embedding_dim,
                num_tasks=num_tasks,
                device=device,
            )

        def _build_actor(self,actor_args):
            n_obs = actor_args['n_obs']
            n_act = actor_args['n_act']
            num_envs = actor_args['num_envs']
            init_scale = actor_args['init_scale']
            std_min = actor_args['std_min'] 
            std_max = actor_args['std_max']
            actor_hidden_dim = actor_args['hidden_dim']
            learn_task_embedding = actor_args['learn_task_embedding']
            task_embedding_dim = actor_args['task_embedding_dim']
            num_tasks = actor_args['num_tasks']
            device = actor_args['device']

            return Actor(
                n_obs=n_obs,
                n_act=n_act,
                num_envs=num_envs,
                init_scale=init_scale,
                hidden_dim=actor_hidden_dim,
                learn_task_embedding=learn_task_embedding,
                task_embedding_dim=task_embedding_dim,
                num_tasks=num_tasks,
                std_min=std_min,
                std_max=std_max,
                device=device,
            )
        def forward(self, obs_dict):
            """TODO"""
            pass
 
        def is_separate_critic(self):
            return self.separate

        def load(self, params):
            self.actor_hidden_dim = params['actor']['hidden_feature']
            self.init_scale = params['actor']['init_scale']
            self.std_min = params['actor']['std_min']
            self.std_max = params['actor']['std_max']

            self.critic_hidden_dim = params['critic']['hidden_feature']
            self.v_min = params['critic']['v_min']
            self.v_max = params['critic']['v_max']
            self.num_atoms = params['critic']['num_atoms']

            self.has_space = 'space' in params
            self.value_shape = params.get('value_shape', 1)
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            if self.has_space:
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False

# copied from https://github.com/younggyoseo/FastTD3/blob/main/fast_td3/fast_td3.py

class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
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
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        learn_task_embedding: bool,
        task_embedding_dim: int,
        num_tasks: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.qnet1 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.qnet2 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            device=device,
        )

        self.register_buffer(
            "q_support", torch.linspace(v_min, v_max, num_atoms, device=device)
        )

        self.task_embedding = torch.nn.Embedding(
            num_embeddings=num_tasks,
            embedding_dim=task_embedding_dim,
            max_norm=1.0,
        ) if learn_task_embedding else None

        self.num_tasks = num_tasks

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.task_embedding is not None:
            task_ids_one_hot = obs[..., -self.num_tasks :]
            task_indices = torch.argmax(task_ids_one_hot, dim=1)
            task_embeddings = self.task_embedding(task_indices)
            obs = torch.cat([obs[..., : -self.num_tasks], task_embeddings], dim=-1)
        return self.qnet1(obs, actions), self.qnet2(obs, actions)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
    ) -> torch.Tensor:
        """Projection operation that includes q_support directly"""
        if self.task_embedding is not None:
            task_ids_one_hot = obs[..., -self.num_tasks :]
            task_indices = torch.argmax(task_ids_one_hot, dim=1)
            task_embeddings = self.task_embedding(task_indices)
            obs = torch.cat([obs[..., : -self.num_tasks], task_embeddings], dim=-1)

        q1_proj = self.qnet1.projection(
            obs,
            actions,
            rewards,
            bootstrap,
            discount,
            self.q_support,
            self.q_support.device,
        )
        q2_proj = self.qnet2.projection(
            obs,
            actions,
            rewards,
            bootstrap,
            discount,
            self.q_support,
            self.q_support.device,
        )
        return q1_proj, q2_proj

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        return torch.sum(probs * self.q_support, dim=1)


class Actor(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        learn_task_embedding,
        task_embedding_dim: int,
        num_tasks: int,
        std_min: float = 0.05,
        std_max: float = 0.8,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 4, n_act, device=device),
            nn.Tanh(),
        )
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)

        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs

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
        x = self.net(obs)
        action = self.fc_mu(x)
        return action

    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
    ) -> torch.Tensor:
        # If dones is provided, resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            # Generate new noise scales for done environments (one per environment)
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales = torch.where(dones_view, new_scales, self.noise_scales)

        act = self(obs)
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales
        return act + noise