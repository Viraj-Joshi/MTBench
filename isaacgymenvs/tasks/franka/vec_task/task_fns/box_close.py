import os
from typing import Dict, Tuple

import torch
import numpy as np
from gym import spaces

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch
from isaacgymenvs.tasks.reward_utils import _gripper_caging_reward, tolerance, hamacher_product
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
    box_top_asset_file = "assets_v2/unified_objects/box_close_box_top.xml"
    box_base_asset_file = "assets_v2/unified_objects/box_close_box.xml"

    # create box_top asset
    box_top_asset_options = gymapi.AssetOptions()
    box_top_asset_options.fix_base_link = False
    box_top_asset_options.disable_gravity = False
    box_top_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    box_top_asset = self.gym.load_asset(self.sim, asset_root, box_top_asset_file, box_top_asset_options )

    # create box_base asset
    box_base_asset_options = gymapi.AssetOptions()
    box_base_asset_options.fix_base_link = True
    box_base_asset_options.disable_gravity = False
    box_base_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    box_base_asset = self.gym.load_asset(self.sim, asset_root, box_base_asset_file, box_base_asset_options )

    # define start pose for obj (will be reset later)
    obj_height = .04
    obj_start_pose = gymapi.Transform()
    obj_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+obj_height/2)
    obj_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # define start pose for box_top (will be reset later)
    box_top_start_pose = gymapi.Transform()
    box_top_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2])
    box_top_start_pose.r = gymapi.Quat(0, 0, 0, 1)

    # define start pose for box_base (will be reset)
    box_base_start_pose = gymapi.Transform()
    box_base_start_pose.p = gymapi.Vec3(.15, 0, self._table_surface_pos[2])
    box_base_start_pose.r = gymapi.Quat(0, 0, 0, 1 )
    
    
    # ---------------------- Compute aggregate size ----------------------
    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    
    num_object_bodies = self.gym.get_asset_rigid_body_count(box_top_asset) + self.gym.get_asset_rigid_body_count(box_base_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(box_top_asset) + self.gym.get_asset_rigid_shape_count(box_base_asset)
    num_object_dofs = self.gym.get_asset_dof_count(box_top_asset) + self.gym.get_asset_dof_count(box_base_asset)
    
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    print("num object bodies: ", num_object_bodies)
    print("num object dofs: ", num_object_dofs)
    
    
    # ---------------------- Define goals ----------------------
    # obj is used by box_top, goal for target pos and box_base
    obj_low =  (-0.15, -0.05, self._table_surface_pos[2])
    obj_high = (-0.10,  0.05, self._table_surface_pos[2])

    # z matches where lid_base (top_link) sits when the lid rests on the box
    # walls: box wall top is at table + 0.06 and the lid's bottom rim extends
    # 0.016 below top_link, so seated top_link_z = table_surface + 0.076.
    goal_low =  (.15, -.10, self._table_surface_pos[2] + .076)
    goal_high = (.25,  .10, self._table_surface_pos[2] + .076)

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
        
        # Create box_top
        box_top_actor = self.gym.create_actor(env_ptr, box_top_asset, box_top_start_pose, "box_top", i, -1, 0)

        # create box_base
        box_base_actor = self.gym.create_actor(env_ptr, box_base_asset, box_base_start_pose, "box_base", i, -1, 0)
        
        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                    i, 1, 0)

        # if self.aggregate_mode > 0:
        #     self.gym.end_aggregate(env_ptr)

        envs.append(env_ptr)
        frankas.append(franka_actor)
        objects.append(box_top_actor)

    # pass these to the main create envs fns
    num_task_actor = 2  # box_top and box_base
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    return envs, frankas, objects, random_reset_space, num_task_actor, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env
    
    lid_base_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 1
    lid_base_rigid_body_states = self.rigid_body_states[lid_base_rigid_body_idx].view(-1, 13)

    # lid gripping site
    obj_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 2
    obj_rigid_body_states = self.rigid_body_states[obj_rigid_body_idx].view(-1, 13)
    obj_pos = obj_rigid_body_states[:, 0:3] - to_torch([0,0,.01],device=self.device)
    obj_rot = obj_rigid_body_states[:, 3:7]

    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    box_base_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    box_base_quat = self.root_state_tensor[multi_env_ids_int32,3:7]

    self.specialized_kwargs['box_close']['lid_base_pos'] = lid_base_rigid_body_states[:, 0:3]

    return torch.cat([
        obj_pos,
        obj_rot,
        box_base_pos,
        box_base_quat
    ], dim=-1)

@torch.jit.script
def _reward_quat(obj_rot: torch.Tensor) -> torch.Tensor:
    # Ideal upright lid has quat [0, 0, 0, 1]
    # Rather than deal with an angle between quaternions, just approximate:
    error = obj_rot.clone()
    error[:,-1] += -1
    error = torch.norm(error,dim=-1)
    return torch.maximum(1.0 - error / 0.2, torch.zeros_like(error))

@torch.jit.script
def compute_reward(
    reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, franka_dof_pos: torch.Tensor,
    franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, max_episode_length: float,
    tcp_init: torch.Tensor, target_pos: torch.Tensor, lid: torch.Tensor, obj_init_pos: torch.Tensor, specialized_kwargs: Dict[str,torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    lid_base_pos = specialized_kwargs["lid_base_pos"]

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2
    tcp_to_lid = torch.norm(tcp - lid, dim=-1)

    # Cage the lid's handle between the gripper fingers.
    object_grasped = _gripper_caging_reward(
        lid,
        franka_lfinger_pos,
        franka_rfinger_pos,
        tcp,
        tcp_init,
        actions,
        obj_init_pos,
        object_reach_radius=0.01,
        obj_radius=0.02,
        pad_success_thresh=0.05,
        xz_thresh=0.005,
        medium_density=True,
    )

    # Place lid_base (the lid's base frame, which actually seats on the box)
    # onto target_pos.
    lid_base_to_target = torch.norm(lid_base_pos - target_pos, dim=-1)
    in_place_margin = torch.norm(obj_init_pos - target_pos, dim=-1)
    in_place = tolerance(
        lid_base_to_target,
        bounds=(0.0, 0.02),
        margin=in_place_margin,
        sigmoid="long_tail",
    )

    # Gripper openness: 1 ≈ fully open, 0 ≈ fully closed.
    finger1_dof = franka_dof_pos[:, -2]
    finger2_dof = -franka_dof_pos[:, -1]
    tcp_opened = (finger1_dof - finger2_dof) / .08

    in_place_and_grasped = hamacher_product(object_grasped, in_place)
    rewards = in_place_and_grasped

    # Big bonus only when the handle is caged AND the lid is off the table.
    lifted_mask = (tcp_to_lid < 0.02) & (tcp_opened > 0) & ((lid[:, 2] - 0.01) > obj_init_pos[:, 2])
    rewards = torch.where(lifted_mask, rewards + 1.0 + 5.0 * in_place, rewards) * 0.01

    success = lid_base_to_target < 0.05
    rewards = torch.where(success, 10, rewards)

    # Compute resets
    reset_buf = torch.logical_or(success, progress_buf >= max_episode_length - 1).to(reset_buf.dtype)

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

    self.obj_init_pos[env_ids] =  self.last_rand_vecs[env_ids,:3] + to_torch([0,0,.106],device=self.device) # should be .09 but for some reason it is .106
    self.target_pos[env_ids] = self.last_rand_vecs[env_ids,3:]

    # reset franka
    pos = tensor_clamp(
        self.franka_default_dof_pos[tid].unsqueeze(0) + self.reset_noise * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - .5),
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
    
    # self.dof_targets_all[self.franka_dof_idx].view(self.num_envs,-1)[env_ids, :self.num_franka_dofs] = pos # this doesnt work because advanced indexing creates a copy
    self.dof_targets_all.flatten().index_copy_(0,reset_franka_dof_idx[env_ids].flatten(),all_pos[env_ids].flatten())

    # reset effort control to 0
    self.effort_control_all.flatten().index_copy_(0,reset_franka_dof_idx[env_ids].flatten(),torch.zeros_like(self._effort_control[env_ids]).flatten())
    
    # reset object pose
    box_base_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    self.root_state_tensor[box_base_multi_env_ids_int32,:3] = torch.vstack((self.target_pos[env_ids,0],self.target_pos[env_ids,1],\
                                                                            torch.ones_like(self.target_pos[env_ids,0])*self._table_surface_pos[2])).T

    box_top_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    self.root_state_tensor[box_top_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,:3]
    self.root_state_tensor[box_top_multi_env_ids_int32,3:7] = to_torch([0,0,0,1],device=self.device)
    self.root_state_tensor[box_top_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    actor_multi_env_ids_int32 = torch.cat((box_base_multi_env_ids_int32,box_top_multi_env_ids_int32),dim=-1).flatten().to(dtype=torch.int32)

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return self.franka_actor_idx[env_ids], actor_multi_env_ids_int32

