import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from rl_games.algos_torch import network_builder
from isaacgymenvs.learning.networks.simba_v2 import Actor, Critic

from typing import List, Tuple

class SimbaV2A2CBuilder(network_builder.NetworkBuilder):
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
            task_embedding_dim = kwargs.pop('task_embedding_dim',0)
            # get the dim of the real part of the obs
            real_obs_dim = obs_shape[0] - task_embedding_dim

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            actor_args = {
                'n_obs' : obs_shape[0],
                'n_act' : action_shape[0],
                'num_envs' : num_envs,
                'init_scale' : self.init_scale,
                'std_min' : self.std_min,
                'std_max' : self.std_max,
                'hidden_dim' : self.actor_units,
                'scaler_init' : self.actor_scaler_init,
                'scaler_scale' : self.actor_scaler_scale,
                'alpha_init' : self.actor_alpha_init,
                'alpha_scale' : self.actor_alpha_scale,
                'expansion' : self.actor_expansion,
                'c_shift' : self.actor_c_shift,
                'num_blocks' : self.actor_num_blocks,
                'device' : device,
            }

            critic_args = {
                'n_obs' : obs_shape[0],
                'n_act' : action_shape[0],
                'num_atoms' : self.num_atoms,
                'v_min' : self.v_min,
                'v_max' : self.v_max,
                'hidden_dim' : self.critic_units,
                'scaler_init' : self.critic_scaler_init,
                'scaler_scale' : self.critic_scaler_scale,
                'alpha_init' : self.critic_alpha_init,
                'alpha_scale' : self.critic_alpha_scale,
                'expansion' : self.critic_expansion,
                'c_shift' : self.critic_c_shift,
                'num_blocks' : self.critic_num_blocks,
                'device' : device,
            }
            
            print("Building Actor")
            self.actor = self._build_actor(actor_args)

            print("Building Critic")
            self.critic = self._build_critic(critic_args)

        def _build_critic(self, critic_args):
            n_obs = critic_args['n_obs']
            n_act = critic_args['n_act']
            num_atoms = critic_args['num_atoms']
            v_min = critic_args['v_min']
            v_max = critic_args['v_max']
            critic_hidden_dim = critic_args['hidden_dim']
            # Simba Specific parameters
            critic_scaler_init = critic_args['scaler_init']
            critic_scaler_scale = critic_args['scaler_scale']
            critic_alpha_init = critic_args['alpha_init']
            critic_alpha_scale = critic_args['alpha_scale']
            critic_expansion = critic_args['expansion']
            critic_c_shift = critic_args['c_shift']
            critic_num_blocks = critic_args['num_blocks']
            device = critic_args['device']

            return Critic(
                n_obs=n_obs,
                n_act=n_act,
                num_atoms=num_atoms,
                v_min=v_min,
                v_max=v_max,
                hidden_dim=critic_hidden_dim,
                scaler_init=critic_scaler_init,
                scaler_scale=critic_scaler_scale,
                alpha_init=critic_alpha_init,
                alpha_scale=critic_alpha_scale,
                expansion=critic_expansion,
                c_shift=critic_c_shift,
                num_blocks=critic_num_blocks,
                device=device,
            )

        def _build_actor(self,actor_args):
            n_obs = actor_args['n_obs']
            n_act = actor_args['n_act']
            num_envs = actor_args['num_envs']
            init_scale = actor_args['init_scale']
            std_min = actor_args['std_min'] 
            std_max = actor_args['std_max']
            actor_hidden_dim = actor_args['units']
            # Simba Specific parameters
            scaler_init = actor_args.get['scaler_init']
            scaler_scale = actor_args.get['scaler_scale']
            alpha_init = actor_args.get['alpha_init']
            alpha_scale = actor_args.get['alpha_scale']
            expansion = actor_args.get['expansion']
            c_shift = actor_args.get['c_shift']
            actor_num_blocks = actor_args['num_blocks']
            device = actor_args['device']

            return Actor(
                n_obs=n_obs,
                n_act=n_act,
                num_envs=num_envs,
                init_scale=init_scale,
                hidden_dim=actor_hidden_dim,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                alpha_init=alpha_init,
                alpha_scale=alpha_scale,
                expansion=expansion,
                c_shift=c_shift,
                num_blocks=actor_num_blocks,
                std_min=std_min,
                std_max=std_max,
                device=device,
            )
        def forward(self, obs_dict):
            """TODO"""
            obs = obs_dict['obs']
            latent_obs = self.state_encoder(obs)
            latent_task_embedding = self.task_encoder(self.task_embedding)
            mu, sigma = self.actor(latent_obs)
            return mu, sigma
 
        def is_separate_critic(self):
            return self.separate

        def load(self, params):
            self.actor_hidden_dim = params['actor']['hidden_dim']
            self.actor_num_blocks = params['actor']['num_blocks']
            self.init_scale = params['actor']['init_scale']
            self.std_min = params['actor']['std_min']
            self.std_max = params['actor']['std_max']

            self.actor_scaler_init = math.sqrt(2.0 / self.actor_hidden_dim)
            self.actor_scaler_scale = math.sqrt(2.0 / self.actor_hidden_dim)
            self.actor_alpha_init = 1.0 / (self.actor_num_blocks + 1)
            self.actor_alpha_scale = 1.0 / math.sqrt(self.actor_hidden_dim)
            self.actor_expansion = 4
            self.actor_c_shift = 3.0

            self.critic_hidden_dim = params['critic']['hidden_dim']
            self.critic_num_blocks = params['critic']['num_blocks']
            self.v_min = params['critic']['v_min']
            self.v_max = params['critic']['v_max']
            self.num_atoms = params['critic']['num_atoms']

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
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False

