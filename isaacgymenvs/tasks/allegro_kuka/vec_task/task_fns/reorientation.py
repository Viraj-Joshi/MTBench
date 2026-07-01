"""Reorientation task hooks for the multi-task Allegro-Kuka env.

Lifted from ``allegro_kuka/allegro_kuka_reorientation.py`` (SAPG).  The goal is a
separate cube that must be matched in both position and orientation, so four
corner keypoints are used.  The object is NOT reset when only the target changes
(this enables consecutive successes within one episode).
"""

import os
from typing import List

import torch
from isaacgym import gymapi
from torch import Tensor

from isaacgymenvs.utils.torch_jit_utils import to_torch, torch_rand_float


def object_keypoint_offsets() -> List[List[float]]:
    return [
        [1, 1, 1],
        [1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ]


def load_additional_assets(env, object_asset_root, arm_pose):
    object_asset_options = gymapi.AssetOptions()
    object_asset_options.disable_gravity = True
    env.reorientation_goal_assets = []
    # the goal cubes must match this task's own object pool (set by the orchestrator)
    for object_asset_file in env._loading_task_files:
        object_asset_dir = os.path.dirname(object_asset_file)
        object_asset_fname = os.path.basename(object_asset_file)
        goal_asset_ = env.gym.load_asset(env.sim, object_asset_dir, object_asset_fname, object_asset_options)
        env.reorientation_goal_assets.append(goal_asset_)
    goal_rb_count = env.gym.get_asset_rigid_body_count(env.reorientation_goal_assets[0])
    goal_shapes_count = env.gym.get_asset_rigid_shape_count(env.reorientation_goal_assets[0])
    return goal_rb_count, goal_shapes_count


def create_additional_objects(env, env_ptr, env_idx, object_asset_idx):
    goal_displacement = gymapi.Vec3(-0.35, -0.06, 0.12)
    goal_start_pose = gymapi.Transform()
    goal_start_pose.p = env.object_start_pose.p + goal_displacement
    goal_start_pose.p.z -= 0.04

    goal_asset = env.reorientation_goal_assets[object_asset_idx]
    goal_handle = env.gym.create_actor(
        env_ptr, goal_asset, goal_start_pose, "goal_object", env_idx + env.num_envs, 0, 0
    )
    goal_object_idx = env.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
    env.goal_object_indices.append(goal_object_idx)
    if env_idx == 0:
        for name in env.gym.get_actor_rigid_body_names(env_ptr, goal_handle):
            env.rigid_body_name_to_idx["goal/" + name] = env.gym.find_actor_rigid_body_index(
                env_ptr, goal_handle, name, gymapi.DOMAIN_ENV
            )
    if env.object_type != "block":
        env.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))


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

        new_rot = env.get_random_quat(env_ids)
        env.goal_states[env_ids, 3:7] = new_rot
        env.root_state_tensor[env.goal_object_indices[env_ids], 3:7] = env.goal_states[env_ids, 3:7]
        env.root_state_tensor[env.goal_object_indices[env_ids], 7:13] = torch.zeros_like(
            env.root_state_tensor[env.goal_object_indices[env_ids], 7:13]
        )

    env.deferred_set_actor_root_state_tensor_indexed([env.goal_object_indices[env_ids]])


def extra_reset_rules(env, resets, task_mask: Tensor):
    """Reset reorientation envs whose hand drifted too far from the object."""
    far = env.curr_fingertip_distances.max(dim=-1).values > 1.5
    resets = torch.where(task_mask & far, torch.ones_like(env.reset_buf), resets)
    return resets
