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
    block_asset_file = "assets_v2/unified_objects/block.xml"
    table_asset_file = "assets_v2/unified_objects/table_with_hole.xml"

    # create block asset
    block_asset_options = gymapi.AssetOptions()
    block_asset_options.fix_base_link = False
    block_asset_options.disable_gravity = False
    block_asset = self.gym.load_asset(self.sim, asset_root, block_asset_file, block_asset_options)
    # block_asset = self.gym.create_box(self.sim, 0.02, 0.02, 0.02, block_asset_options)

    table_pos = [0.0, 0.0, .872+.128]
    table_thickness = 0.065
    
    table_with_hole_opts = gymapi.AssetOptions()
    table_with_hole_opts.fix_base_link = True
    table_with_hole_opts.use_mesh_materials = True
    table_with_hole_opts.collapse_fixed_joints = False
    table_with_hole_asset = self.gym.load_asset(self.sim, asset_root, table_asset_file, table_with_hole_opts)

    # Create table stand asset
    table_stand_height = 0.01
    table_stand_pos = [-.6, 0, table_pos[2] + table_thickness / 2 + table_stand_height / 2]
    table_stand_opts = gymapi.AssetOptions()
    table_stand_opts.fix_base_link = True
    table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_stand_opts)

    # override start pose for franka
    franka_start_pose = gymapi.Transform()
    franka_start_pose.p = gymapi.Vec3(-0.55, 0.0, table_pos[2] + table_thickness / 2 + table_stand_height)
    franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # table surface pos is different for this environment
    _table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])

    # Define start pose for table stand
    table_stand_start_pose = gymapi.Transform()
    table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
    table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # Override start pose for table with hole stand
    table_with_hole_start_pose = gymapi.Transform()
    table_with_hole_start_pose.p = gymapi.Vec3(*table_pos)
    table_with_hole_start_pose.r = gymapi.Quat(0, 0, -0.7071068, 0.7071068 )

    # define start pose for block (will be reset later)
    block_height = 0.04
    block_start_pose = gymapi.Transform()
    block_start_pose.p = gymapi.Vec3(.3, 0, _table_surface_pos[2]+block_height/2)
    block_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    # ---------------------- Compute aggregate size ----------------------
    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    
    num_object_bodies = self.gym.get_asset_rigid_body_count(block_asset)  
    num_object_shapes = self.gym.get_asset_rigid_shape_count(block_asset) 
    num_object_dofs = self.gym.get_asset_dof_count(block_asset)

    num_table_with_hole_rigid_bodies = self.gym.get_asset_rigid_body_count(table_with_hole_asset)
    num_table_with_hole_rigid_shapes = self.gym.get_asset_rigid_shape_count(table_with_hole_asset)
    
    max_agg_bodies = num_franka_bodies + num_object_bodies + num_table_with_hole_rigid_bodies + 1
    max_agg_shapes = num_franka_shapes + num_object_shapes + num_table_with_hole_rigid_shapes + 1

    print("num object bodies: ", num_object_bodies)
    print("num object dofs: ", num_object_dofs)
    
    # ---------------------- Define goals ----------------------
    # obj is for cube and goal is for target
    obj_low =  (0.0, -.1, _table_surface_pos[2]+block_height/2)
    obj_high = (0.1,  .1, _table_surface_pos[2]+block_height/2)

    goal_low =  (.20, -.04, _table_surface_pos[2]-.02)
    goal_high = (.28,  .04, _table_surface_pos[2]-.02)

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
        
        # Create block
        block_actor = self.gym.create_actor(env_ptr, block_asset, block_start_pose, "block", i, -1, 0)
        # self.gym.set_rigid_body_color(env_ptr, block_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))
        
        # override with table with hole
        # table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
        table_with_hole_actor = self.gym.create_actor(env_ptr, table_with_hole_asset, table_with_hole_start_pose, "table_with_hole",
                                                    i, -1, 0)
        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                    i, 1, 0)
    
        # if self.aggregate_mode > 0:
        #     self.gym.end_aggregate(env_ptr)

        envs.append(env_ptr)
        frankas.append(franka_actor)
        objects.append(block_actor)

    # pass these to the main create envs fns
    num_task_actor = 1  # block
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    return envs, frankas, objects, random_reset_space, num_task_actor, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env
    
    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    obj_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    obj_rot = self.root_state_tensor[multi_env_ids_int32,3:7]
    
    return torch.cat([
        obj_pos,
        obj_rot,
        torch.zeros_like(obj_pos),
        torch.zeros_like(obj_rot),
    ], dim=-1)

@torch.jit.script
def compute_reward(
        reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, franka_dof_pos: torch.Tensor,
        franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, max_episode_length: float, 
        init_tcp: torch.Tensor, target_pos: torch.Tensor, obj_pos: torch.Tensor, obj_init_pos: torch.Tensor, specialized_kwargs: Dict[str,torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    TARGET_RADIUS = 0.05

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2

    target_to_obj = torch.norm(obj_pos - target_pos,dim=-1)
    target_to_obj_init = torch.norm(obj_init_pos - target_pos,dim=-1)

    in_place = tolerance(
        target_to_obj,
        bounds=(0.0, TARGET_RADIUS),
        margin=target_to_obj_init ,
        sigmoid="long_tail",
    )

    object_grasped = _gripper_caging_reward(
        obj_pos,
        franka_lfinger_pos,
        franka_rfinger_pos,
        tcp,
        init_tcp,
        actions,
        obj_init_pos,
        object_reach_radius=0.0,
        obj_radius=0.02,
        pad_success_thresh=0.05,
        xz_thresh=0.005,
        medium_density=True
    )

    rewards = hamacher_product(object_grasped, in_place)

    gripper_distance_apart = torch.norm(franka_rfinger_pos-franka_lfinger_pos,dim=-1)
    tcp_opened = torch.clip(gripper_distance_apart/.095,0.0,1.0)

    tcp_to_obj = torch.norm(obj_pos - tcp, dim=-1)
    tcp_to_target = torch.norm(target_pos - tcp, dim=-1)
    rewards = torch.where((tcp_to_obj < 0.02) & (tcp_opened > 0) & (tcp_to_target < TARGET_RADIUS), rewards + 1.0 + 7.0 * in_place, rewards) * .01

    success = (target_to_obj < TARGET_RADIUS)
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

    self.target_pos[env_ids] = self.last_rand_vecs[env_ids, 3:]
    self.obj_init_pos[env_ids] =  self.last_rand_vecs[env_ids, :3]

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
    
    # self.dof_targets_all[self.franka_dof_idx].view(self.num_envs,-1)[env_ids, :self.num_franka_dofs] = pos # this doesnt work because advanced indexing creates a copy
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

    return self.franka_actor_idx[env_ids],obj_multi_env_ids_int32

