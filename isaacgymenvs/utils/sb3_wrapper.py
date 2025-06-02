import isaacgym
from isaacgymenvs.tasks.base.vec_task import VecTask

import hydra
from omegaconf import DictConfig, OmegaConf

import gymnasium as gym
import numpy as np
import torch
from typing import Any, Dict, List

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn


class Sb3VecEnvWrapper(VecEnv):
    
    def __init__(self, env: VecTask):
        self.env = env
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=env.observation_space.shape, dtype=env.observation_space.dtype)
        action_space = gym.spaces.Box(low=-1, high=1, shape=env.action_space.shape, dtype=env.action_space.dtype)
        VecEnv.__init__(self, self.env.num_envs, observation_space, action_space)
        self._ep_rew_buf = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)
        self._ep_len_buf = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)
        
    def get_episode_rewards(self) -> List[float]:
        """Returns the rewards of all the episodes."""
        return self._ep_rew_buf.cpu().tolist()

    def get_episode_lengths(self) -> List[int]:
        """Returns the number of time-steps of all the episodes."""
        return self._ep_len_buf.cpu().tolist()
    
    def reset(self) -> VecEnvObs:  # noqa: D102
        obs_dict = self.env.reset()
        # convert data types to numpy depending on backend
        return self._process_obs(obs_dict)
    
    def step(self, actions: np.ndarray) -> VecEnvStepReturn:  # noqa: D102
        # convert input to numpy array
        actions = np.asarray(actions)
        # convert to tensor
        actions = torch.from_numpy(actions).to(device=self.env.device)
        # record step information
        obs_dict, rew, dones, extras = self.env.step(actions)

        # update episode un-discounted return and length
        self._ep_rew_buf += rew
        self._ep_len_buf += 1
        reset_ids = (dones > 0).nonzero(as_tuple=False)

        # convert data types to numpy depending on backend
        # Note: IsaacEnv uses torch backend (by default).
        obs = self._process_obs(obs_dict)
        rew = rew.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()
        # convert extra information to list of dicts
        infos = self._process_extras(obs, dones, extras, reset_ids)

        # reset info for terminated environments
        self._ep_rew_buf[reset_ids] = 0
        self._ep_len_buf[reset_ids] = 0

        return obs, rew, dones, infos

    def close(self):  # noqa: D102
        pass
        
    """
    Unused methods.
    """

    def step_async(self, actions):  # noqa: D102
        self._async_actions = actions

    def step_wait(self):  # noqa: D102
        return self.step(self._async_actions)

    def get_attr(self, attr_name, indices=None):  # noqa: D102
        getattr(self.env, attr_name)

    def set_attr(self, attr_name, value, indices=None):  # noqa: D102
        raise NotImplementedError

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):  # noqa: D102
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):  # noqa: D102
        raise NotImplementedError

    def get_images(self):  # noqa: D102
        raise NotImplementedError
    
    """
    Helper functions.
    """

    def _process_obs(self, obs_dict) -> np.ndarray:
        """Convert observations into NumPy data type."""
        # Sb3 doesn't support asymmetric observation spaces, so we only use "policy"
        obs = obs_dict["obs"]
        obs = obs.detach().cpu().numpy()
        return obs
    
    def _process_extras(self, obs, dones, extras, reset_ids) -> List[Dict[str, Any]]:
            """Convert miscellaneous information into dictionary for each sub-environment."""
            # create empty list of dictionaries to fill
            infos: List[Dict[str, Any]] = [dict.fromkeys(extras.keys()) for _ in range(self.env.num_envs)]
            # fill-in information for each sub-environment
            # Note: This loop becomes slow when number of environments is large.
            for idx in range(self.env.num_envs):
                # fill-in episode monitoring info
                if idx in reset_ids:
                    infos[idx]["episode"] = dict()
                    infos[idx]["episode"]["r"] = float(self._ep_rew_buf[idx])
                    infos[idx]["episode"]["l"] = float(self._ep_len_buf[idx])
                else:
                    infos[idx]["episode"] = None
                # fill-in information from extras
                for key, value in extras.items():
                    # 1. remap the key for time-outs for what SB3 expects
                    # 2. remap extra episodes information safely
                    # 3. for others just store their values
                    if key == "time_outs":
                        infos[idx]["TimeLimit.truncated"] = bool(value[idx])
                    elif key == "episode":
                        # only log this data for episodes that are terminated
                        if infos[idx]["episode"] is not None:
                            for sub_key, sub_value in value.items():
                                infos[idx]["episode"][sub_key] = sub_value
                    else:
                        if isinstance(value, list):
                            infos[idx][key] = value[idx]
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                infos[idx][key + sub_key] = sub_value[idx]
                # add information about terminal observation separately
                if dones[idx] == 1:
                    # extract terminal observations
                    if isinstance(obs, dict):
                        terminal_obs = dict.fromkeys(obs.keys())
                        for key, value in obs.items():
                            terminal_obs[key] = value[idx]
                    else:
                        terminal_obs = obs[idx]
                    # add info to dict
                    infos[idx]["terminal_observation"] = terminal_obs
                else:
                    infos[idx]["terminal_observation"] = None
            # return list of dictionaries
            return infos
        
        
@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    import logging
    import os
    from datetime import datetime

    # noinspection PyUnresolvedReferences
    import isaacgym
    from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
    from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
    from hydra.utils import to_absolute_path
    from isaacgymenvs.tasks import isaacgym_task_map
    import gym
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs


    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    task_config = omegaconf_to_dict(cfg.task)
    task_name = "KukaReaching"
    
    cuda = True
    
    # create native task and pass custom config
    envs = isaacgymenvs.make(
        seed=1,
        task=task_name,
        num_envs=8192,
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0 if torch.cuda.is_available() and cuda else -1,
        headless=False if torch.cuda.is_available() and cuda else True,
        multi_gpu=False,
        virtual_screen_capture=False,
        force_render=False,  # if False, no viewer rendering will happen
    )
    
    sb3Env = Sb3VecEnvWrapper(envs)
        
if __name__ == "__main__":
    launch_rlg_hydra()