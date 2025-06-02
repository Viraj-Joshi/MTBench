import os
from typing import Dict, Tuple

import torch
import numpy as np
from gym import spaces

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch
from isaacgymenvs.tasks.reward_utils import tolerance, hamacher_product
from isaacgymenvs.utils.torch_jit_utils import tensor_clamp, to_torch


def create_envs(
        env, num_envs, spacing, num_per_row,
        franka_asset, franka_start_pose, franka_dof_props,
        table_asset, table_start_pose, table_stand_asset, table_stand_start_pose,
    ):
    self = env
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)

    # ---------------------- Load assets ----------------------
    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../assets")
    cylinder_asset_file = "assets_v2/unified_objects/cylinder.xml"
    wall_asset_file = "assets_v2/unified_objects/push_wall.xml"

    # create cylinder asset
    cylinder_asset_options = gymapi.AssetOptions()
    cylinder_asset_options.fix_base_link = False
    cylinder_asset_options.disable_gravity = False
    cylinder_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    cylinder_asset = self.gym.load_asset(self.sim, asset_root, cylinder_asset_file, cylinder_asset_options )

    # create wall asset
    wall_asset_options = gymapi.AssetOptions()
    wall_asset_options.fix_base_link = True
    wall_asset_options.disable_gravity = False
    wall_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_file, wall_asset_options )

    # define start pose for cylinder (will be reset later)
    cylinder_height = .04
    cylinder_start_pose = gymapi.Transform()
    cylinder_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+cylinder_height/2)
    cylinder_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # define start pose for wall (will NOT be reset)
    wall_height = .06
    wall_start_pose = gymapi.Transform()
    wall_start_pose.p = gymapi.Vec3(.15, 0, self._table_surface_pos[2]+wall_height/2)
    wall_start_pose.r = gymapi.Quat(0, 0, 0.7071068, 0.7071068 )
    
    
    # ---------------------- Compute aggregate size ----------------------
    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    
    num_object_bodies = self.gym.get_asset_rigid_body_count(cylinder_asset) + self.gym.get_asset_rigid_body_count(wall_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(cylinder_asset) + self.gym.get_asset_rigid_shape_count(wall_asset)
    num_object_dofs = self.gym.get_asset_dof_count(cylinder_asset) + self.gym.get_asset_dof_count(wall_asset)
    
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    print("num object bodies: ", num_object_bodies)
    print("num object dofs: ", num_object_dofs)
    
    
    # obj is used by cylinder and goal for target pos
    obj_low = (0, -.05, self._table_surface_pos[2]+cylinder_height/2)
    obj_high = (.05, .05, self._table_surface_pos[2]+cylinder_height/2)

    goal_low = (.25, -.05, self._table_surface_pos[2]+.05)
    goal_high = (.3, .05, self._table_surface_pos[2]+.3)

    # goal_space = spaces.Box(np.array(goal_low),np.array(goal_high))
    random_reset_space = spaces.Box(
        np.hstack((obj_low, goal_low)),
        np.hstack((obj_high, goal_high)),
    )
    
    
    # ---------------------- Create envs ----------------------
    envs = []
    frankas = []
    objects = []

    for i in range(num_envs):
        # create env instance
        env_ptr = self.gym.create_env(
            self.sim, lower, upper, num_per_row
        )

        # if self.aggregate_mode >= 3:
        #     self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

        franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
        self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

        if self.aggregate_mode == 2:
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

        if self.aggregate_mode == 1:
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
        
        # Create cylinder
        cylinder_actor = self.gym.create_actor(env_ptr, cylinder_asset, cylinder_start_pose, "cylinder", i, -1, 0)

        # create wall
        wall_actor = self.gym.create_actor(env_ptr, wall_asset, wall_start_pose, "wall", i, -1, 0)
        
        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                    i, 1, 0)

        # if self.aggregate_mode > 0:
        #     self.gym.end_aggregate(env_ptr)

        envs.append(env_ptr)
        frankas.append(franka_actor)
        objects.append(cylinder_actor)

    # pass these to the main create envs fns
    num_task_actor = 2  # cylinder and wall
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    return envs, frankas, objects, random_reset_space, num_task_actor, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env
    
    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    obj_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    obj_rot = self.root_state_tensor[multi_env_ids_int32,3:7]

    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    wall_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    wall_rot = self.root_state_tensor[multi_env_ids_int32,3:7]
    
    return torch.cat([
        obj_pos,
        obj_rot,
        wall_pos,
        wall_rot
    ], dim=-1)


@torch.jit.script
def compute_reward(
    reset_buf, progress_buf, actions, franka_dof_pos, franka_lfinger_pos, franka_rfinger_pos, max_episode_length,
    hand_init_pos, target_pos, obj_pos, obj_init_pos, specialized_kwargs
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Dict[str,Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    TARGET_RADIUS = 0.05

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2

    tcp_to_target = torch.norm(tcp - target_pos, dim=-1)

    in_place_margin = torch.norm(hand_init_pos - target_pos, dim=-1)

    in_place = tolerance(
        tcp_to_target,
        bounds=(0.0, TARGET_RADIUS),
        margin = in_place_margin,
        sigmoid='long_tail',
    )

    rewards = in_place * .01
    success = tcp_to_target <= .05

    rewards = torch.where(success, 10, rewards)
    
    # reset upon success or max_iterations reached
    reset_buf = torch.where(success | (progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf, success

def reset_env(env, tid, env_ids, random_reset_space):
    self = env
    
    last_rand_vecs = to_torch(np.random.uniform(
        random_reset_space.low,
        random_reset_space.high,
        size=(len(env_ids), random_reset_space.low.size),
    ).astype(np.float64), device=self.device)

    if torch.all(self.last_rand_vecs[env_ids]==0) and self.fixed:
        self.last_rand_vecs[env_ids] = last_rand_vecs
    elif not self.fixed:
        self.last_rand_vecs[env_ids] = last_rand_vecs


    self.target_pos[env_ids] = self.last_rand_vecs[env_ids,3:]
    self.obj_init_pos[env_ids] =  self.last_rand_vecs[env_ids,:3]

    # reset franka
    pos = tensor_clamp(
        self.franka_default_dof_pos[tid].unsqueeze(0) + self.reset_noise * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
        self.franka_dof_lower_limits, self.franka_dof_upper_limits)
    # Overwrite gripper init pos to open(no noise since these are always effort controlled)
    pos[:, -2:] = self.franka_default_dof_pos[tid][-2:]
    
    all_pos = torch.zeros((self.num_envs,self.num_franka_dofs), device=self.device)
    all_pos[env_ids,:] = pos

    reset_franka_dof_idx = self.franka_dof_idx.view(self.num_envs,self.num_franka_dofs).to(dtype=torch.long)
    self.dof_state[...,0].index_copy_(0,reset_franka_dof_idx[env_ids].flatten(),all_pos[env_ids].flatten())
    self.dof_state[...,1].index_copy_(0,reset_franka_dof_idx[env_ids].flatten(),torch.zeros_like(all_pos[env_ids]).flatten())

    self.franka_dof_pos[env_ids, :] = pos
    self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
    
    self.dof_targets_all.flatten().index_copy_(0,reset_franka_dof_idx[env_ids].flatten(),all_pos[env_ids].flatten())

    # reset effort control to 0
    self.effort_control_all.flatten().index_copy_(0,reset_franka_dof_idx[env_ids].flatten(),torch.zeros_like(self._effort_control[env_ids]).flatten())
    
    # reset object pose
    obj_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    self.root_state_tensor[obj_multi_env_ids_int32,:3] = self.obj_init_pos[env_ids]
    self.root_state_tensor[obj_multi_env_ids_int32,3:7] = to_torch([0,0,0,1],device=self.device)
    self.root_state_tensor[obj_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return self.franka_actor_idx[env_ids], obj_multi_env_ids_int32

