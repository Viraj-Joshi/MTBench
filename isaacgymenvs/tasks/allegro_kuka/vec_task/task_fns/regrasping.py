"""Regrasping task hooks for the multi-task Allegro-Kuka env.

Lifted from ``allegro_kuka/allegro_kuka_regrasping.py`` (SAPG), reshaped from a
subclass into module-level functions that take the orchestrator ``env`` as the
first argument.  The goal is a single ball marker; object orientation does not
matter, so a single (centered) keypoint is used.  The object is re-randomized
whenever a new target is sampled.
"""

from typing import List

import torch
from isaacgym import gymapi
from torch import Tensor

from isaacgymenvs.utils.torch_jit_utils import torch_rand_float


def object_keypoint_offsets() -> List[List[float]]:
    # single keypoint -- orientation is irrelevant for regrasping
    return [[0.0, 0.0, 0.0]]


def load_additional_assets(env, object_asset_root, arm_pose):
    goal_asset_options = gymapi.AssetOptions()
    goal_asset_options.disable_gravity = True
    env.regrasping_goal_asset = env.gym.load_asset(
        env.sim, object_asset_root, env.asset_files_dict["ball"], goal_asset_options
    )
    goal_rb_count = env.gym.get_asset_rigid_body_count(env.regrasping_goal_asset)
    goal_shapes_count = env.gym.get_asset_rigid_shape_count(env.regrasping_goal_asset)
    return goal_rb_count, goal_shapes_count


def create_additional_objects(env, env_ptr, env_idx, object_asset_idx):
    goal_start_pose = gymapi.Transform()
    goal_handle = env.gym.create_actor(
        env_ptr, env.regrasping_goal_asset, goal_start_pose, "goal_object", env_idx + env.num_envs, 0, 0
    )
    env.gym.set_actor_scale(env_ptr, goal_handle, 0.5)
    env.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
    goal_object_idx = env.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
    if env_idx == 0:
        for name in env.gym.get_actor_rigid_body_names(env_ptr, goal_handle):
            env.rigid_body_name_to_idx["goal/" + name] = env.gym.find_actor_rigid_body_index(
                env_ptr, goal_handle, name, gymapi.DOMAIN_ENV
            )
    env.goal_object_indices.append(goal_object_idx)


def reset_target(env, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True) -> None:
    if len(env_ids) > 0 and tensor_reset:
        target_volume_origin = env.target_volume_origin
        target_volume_extent = env.target_volume_extent

        target_volume_min_coord = target_volume_origin + target_volume_extent[:, 0]
        target_volume_max_coord = target_volume_origin + target_volume_extent[:, 1]
        target_volume_size = target_volume_max_coord - target_volume_min_coord

        rand_pos_floats = torch_rand_float(0.0, 1.0, (len(env_ids), 3), device=env.device)
        target_coords = target_volume_min_coord + rand_pos_floats * target_volume_size
        env.goal_states[env_ids, 0:3] = target_coords
        env.root_state_tensor[env.goal_object_indices[env_ids], 0:3] = env.goal_states[env_ids, 0:3]

    env.deferred_set_actor_root_state_tensor_indexed([env.goal_object_indices[env_ids]])
    # regrasping also re-randomizes the object whenever the target changes
    env.reset_object_pose(env_ids, reset_buf_idxs, tensor_reset=tensor_reset)
