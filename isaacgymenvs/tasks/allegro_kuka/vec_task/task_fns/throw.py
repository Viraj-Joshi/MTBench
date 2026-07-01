"""Throw task hooks for the multi-task Allegro-Kuka env.

Lifted from ``allegro_kuka/allegro_kuka_throw.py`` (SAPG).  The goal is to throw
the object into a bucket placed to the left or right of the table; object
orientation is irrelevant so a single (centered) keypoint is used.  The extra
"goal" actor for this task is the bucket itself.
"""

from typing import List

import torch
from isaacgym import gymapi
from torch import Tensor

from isaacgymenvs.utils.torch_jit_utils import torch_rand_float


def object_keypoint_offsets() -> List[List[float]]:
    return [[0.0, 0.0, 0.0]]


def load_additional_assets(env, object_asset_root, arm_pose):
    bucket_asset_options = gymapi.AssetOptions()
    bucket_asset_options.disable_gravity = False
    bucket_asset_options.fix_base_link = True
    bucket_asset_options.collapse_fixed_joints = True
    bucket_asset_options.vhacd_enabled = True
    bucket_asset_options.vhacd_params = gymapi.VhacdParams()
    bucket_asset_options.vhacd_params.resolution = 500000
    bucket_asset_options.vhacd_params.max_num_vertices_per_ch = 32
    bucket_asset_options.vhacd_params.min_volume_per_ch = 0.001
    env.throw_bucket_asset = env.gym.load_asset(
        env.sim, object_asset_root, env.asset_files_dict["bucket"], bucket_asset_options
    )

    env.throw_bucket_pose = gymapi.Transform()
    env.throw_bucket_pose.p = gymapi.Vec3()
    env.throw_bucket_pose.p.x = arm_pose.p.x - 0.6
    env.throw_bucket_pose.p.y = arm_pose.p.y - 1
    env.throw_bucket_pose.p.z = arm_pose.p.z + 0.45

    bucket_rb_count = env.gym.get_asset_rigid_body_count(env.throw_bucket_asset)
    bucket_shapes_count = env.gym.get_asset_rigid_shape_count(env.throw_bucket_asset)
    print(f"Bucket rb {bucket_rb_count}, shapes {bucket_shapes_count}")
    return bucket_rb_count, bucket_shapes_count


def create_additional_objects(env, env_ptr, env_idx, object_asset_idx):
    bucket_handle = env.gym.create_actor(
        env_ptr, env.throw_bucket_asset, env.throw_bucket_pose, "bucket_object", env_idx, 0, 0
    )
    bucket_object_idx = env.gym.get_actor_index(env_ptr, bucket_handle, gymapi.DOMAIN_SIM)
    if env_idx == 0:
        for name in env.gym.get_actor_rigid_body_names(env_ptr, bucket_handle):
            env.rigid_body_name_to_idx["bucket/" + name] = env.gym.find_actor_rigid_body_index(
                env_ptr, bucket_handle, name, gymapi.DOMAIN_ENV
            )
    env.goal_object_indices.append(bucket_object_idx)


def reset_target(env, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True) -> None:
    if len(env_ids) > 0 and tensor_reset:
        # whether we place the bucket to the left or to the right of the table
        left_right_random = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=env.device)
        x_pos = torch.where(
            left_right_random > 0, 0.5 * torch.ones_like(left_right_random), -0.5 * torch.ones_like(left_right_random)
        )
        x_pos += torch.sign(left_right_random) * torch_rand_float(0, 0.4, (len(env_ids), 1), device=env.device)
        y_pos = torch_rand_float(-1.0, 0.7, (len(env_ids), 1), device=env.device)
        z_pos = torch_rand_float(0.0, 1.0, (len(env_ids), 1), device=env.device)
        env.root_state_tensor[env.goal_object_indices[env_ids], 0:1] = x_pos
        env.root_state_tensor[env.goal_object_indices[env_ids], 1:2] = y_pos
        env.root_state_tensor[env.goal_object_indices[env_ids], 2:3] = z_pos

        env.goal_states[env_ids, 0:1] = x_pos
        env.goal_states[env_ids, 1:2] = y_pos
        env.goal_states[env_ids, 2:3] = z_pos + 0.05

    # also reset the object to its initial position
    env.reset_object_pose(env_ids, reset_buf_idxs, tensor_reset)
    env.deferred_set_actor_root_state_tensor_indexed([env.goal_object_indices[env_ids]])
