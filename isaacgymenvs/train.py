# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import os
from datetime import datetime

# noinspection PyUnresolvedReferences
import isaacgym

import hydra

from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import DictConfig, OmegaConf
import gym

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed


def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv, VisualRLGPUAlgoObserver
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import mt_a2c_agent, ml_a2c_agent, mt_models, mt_player
    from isaacgymenvs.learning.networks import pq_builder, soft_modularized_pq_builder
    from isaacgymenvs.learning import mt_sac_agent, mt_sac_agent_gradmani
    from isaacgymenvs.learning import pqn_agent
    from isaacgymenvs.learning import grpo_agent
    from isaacgymenvs.learning import td3_agent 

    from isaacgymenvs.learning.networks import soft_modularization_sac_builder
    from isaacgymenvs.learning.networks import soft_modularization_a2c_builder
    from isaacgymenvs.learning.networks import moore_sac_builder
    from isaacgymenvs.learning.networks import moore_a2c_builder
    from isaacgymenvs.learning.networks import care_a2c_builder
    from isaacgymenvs.learning.networks import multihead_a2c_builder
    from isaacgymenvs.learning.networks import asymmetric_a2c_builder
    from isaacgymenvs.learning.networks import paco_a2c_builder
    from isaacgymenvs.learning.networks import grpo_builder
    from isaacgymenvs.learning.networks import td3_builder
    
    import isaacgymenvs

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        envs._freeze_rand_vec = True
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        # if cfg.train.params.config.init_at_random_progress:
        #     import torch
        #     envs.progress_buf = torch.randint_like(envs.progress_buf, high=int(envs.max_episode_length))

        # do a speed test here
        # import time
        # import torch
        # t1 = time.time()
        # for _ in range(1000):
        #     actions = envs.action_space.sample()
        #     actions = torch.tensor(actions).unsqueeze(0).repeat(envs.num_envs, 1)
        #     obs, reward, done, info = envs.step(actions)

        # # 512: Speed test: 40379.02472414758 steps per second
        # # 128: Speed test: 10751.227977312297 steps per second
        # # 4096: Speed test: 175570.21751312562 steps per second
        # t2 = time.time()
        # print(f"Speed test: {1000 * envs.num_envs / (t2 - t1)} steps per second")
        # import ipdb; ipdb.set_trace()
        return envs

    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, 'dict_obs_cls') and ige_env_cls.dict_obs_cls else False

    if dict_cls:
        
        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec['obs'] = {'names': list(actor_net_cfg.inputs.keys()), 'concat': not actor_net_cfg.name == "complex_net", 'space_name': 'observation_space'}
        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec['states'] = {'names': list(critic_net_cfg.inputs.keys()), 'concat': not critic_net_cfg.name == "complex_net", 'space_name': 'state_space'}
        
        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs))
    else:

        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [VisualRLGPUAlgoObserver()]

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('mt_a2c_continuous', lambda **kwargs : mt_a2c_agent.MTA2CAgent(**kwargs))
        runner.algo_factory.register_builder('mt_grpo_continuous', lambda **kwargs : grpo_agent.MTGRPOAgent(**kwargs))
        runner.algo_factory.register_builder('famo_grpo', lambda **kwargs : grpo_agent.FAMOGRPOAgent(**kwargs))
        runner.algo_factory.register_builder('ml_a2c_MAMLPPO', lambda **kwargs : ml_a2c_agent.MLA2CAgent(**kwargs))
        runner.algo_factory.register_builder('ml_a2c_REPTILE', lambda **kwargs : ml_a2c_agent.MLA2CReptileAgent(**kwargs))
        runner.algo_factory.register_builder('q_learning', lambda **kwargs : pqn_agent.PQNAgent(**kwargs))
        runner.algo_factory.register_builder('mt_q_learning', lambda **kwargs : pqn_agent.MTPQNAgent(**kwargs))
        runner.algo_factory.register_builder('cagrad_sac', lambda **kwargs : mt_sac_agent_gradmani.CAGradSAC(**kwargs))
        runner.algo_factory.register_builder('pcgrad_a2c', lambda **kwargs : mt_a2c_agent.PCGradA2CAgent(**kwargs))
        runner.algo_factory.register_builder('cagrad_a2c', lambda **kwargs : mt_a2c_agent.CAGradA2CAgent(**kwargs))
        runner.algo_factory.register_builder('famo_a2c', lambda **kwargs : mt_a2c_agent.FAMOA2CAgent(**kwargs))
        runner.algo_factory.register_builder('mt_sac', lambda **kwargs : mt_sac_agent.MTSACAgent(**kwargs))
        runner.algo_factory.register_builder('mt_sac_soft_modularization', lambda **kwargs : mt_sac_agent.MTSACSoftModularizationAgent(**kwargs))
        runner.algo_factory.register_builder('fast_td3', lambda **kwargs : td3_agent.FastTD3Agent(**kwargs))
        runner.algo_factory.register_builder('mt_fast_td3', lambda **kwargs : td3_agent.MTFastTD3Agent(**kwargs))

        model_builder.register_model('mt_continuous_a2c_logstd', lambda network, **kwargs : mt_models.MTModelA2CContinuousLogStd(network))
        model_builder.register_model('mt_continuous_grpo_logstd', lambda network, **kwargs : mt_models.MTModelGRPOContinuousLogStd(network))
        model_builder.register_model('mt_continuous_sac', lambda network, **kwargs : mt_models.MTModelSACContinuous(network))
        model_builder.register_model('fast_td3', lambda network, **kwargs : mt_models.ModelFastTD3(network))
        model_builder.register_model('mt_fast_td3', lambda network, **kwargs : mt_models.MTModelFastTD3Continuous(network))
        model_builder.register_model('parallel_q', lambda network, **kwargs : mt_models.ModelParallelQ(network))
        model_builder.register_model('mt_parallel_q', lambda network, **kwargs : mt_models.MTModelParallelQ(network))
        
        model_builder.register_network('fast_td3_a2c', lambda **kwargs : td3_builder.FastTD3Builder(**kwargs))
        model_builder.register_network('grpo', lambda **kwargs : grpo_builder.GRPOBuilder())
        model_builder.register_network('soft_modularization_sac', lambda **kwargs : soft_modularization_sac_builder.SoftModularizedSACBuilder())
        model_builder.register_network('soft_modularization_ppo', lambda **kwargs : soft_modularization_a2c_builder.SoftModularizedA2CBuilder())
        model_builder.register_network('moore_sac', lambda **kwargs : moore_sac_builder.MOORESACBuilder())
        model_builder.register_network('moore_a2c', lambda **kwargs : moore_a2c_builder.MOOREA2CBuilder())
        model_builder.register_network('care_a2c', lambda **kwargs : care_a2c_builder.CAREA2CBuilder())
        model_builder.register_network('pqn', lambda **kwargs : pq_builder.PQNBuilder())
        model_builder.register_network('soft_modularization_pq', lambda **kwargs : soft_modularized_pq_builder.SoftModularizedPQBuilder())
        model_builder.register_network('multihead_a2c', lambda **kwargs : multihead_a2c_builder.MultiHeadA2CBuilder(**kwargs))
        model_builder.register_network('asymmetric_a2c', lambda **kwargs : asymmetric_a2c_builder.AsymmetricA2CBuilder(**kwargs))
        model_builder.register_network('paco_a2c', lambda **kwargs : paco_a2c_builder.PACOA2CBuilder(**kwargs))

        runner.player_factory.register_builder('mt_a2c_continuous', lambda **kwargs : mt_player.MTPlayer(**kwargs))
        runner.player_factory.register_builder('mt_a2c_paco', lambda **kwargs : mt_player.MTPlayer(**kwargs))
        runner.player_factory.register_builder('moore_ppo', lambda **kwargs : mt_player.MTPlayer(**kwargs))
        runner.player_factory.register_builder('soft_modularization_ppo', lambda **kwargs : mt_player.MTPlayer(**kwargs))
        runner.player_factory.register_builder('multihead_a2c', lambda **kwargs : mt_player.MTPlayer(**kwargs))
        runner.player_factory.register_builder('mt_grpo_continuous', lambda **kwargs : mt_player.MTPlayer(**kwargs))

        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    if not cfg.test:
        experiment_dir = os.path.join('runs', cfg.train.params.config.name + 
        '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })


if __name__ == "__main__":
    launch_rlg_hydra()
