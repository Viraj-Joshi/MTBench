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
    box_asset_file = "assets_v2/unified_objects/cylinder.xml"
    shelf_asset_file = "assets_v2/unified_objects/shelf.xml"
    shelf_texture_path = "assets_v2/textures/wood1.png"

    # create box asset
    box_asset_options = gymapi.AssetOptions()
    box_asset_options.fix_base_link = False
    box_asset_options.disable_gravity = False
    box_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    box_asset = self.gym.load_asset(self.sim, asset_root, box_asset_file, box_asset_options )

    # create shelf asset
    shelf_asset_options = gymapi.AssetOptions()
    shelf_asset_options.fix_base_link = True
    shelf_asset_options.disable_gravity = False
    shelf_asset = self.gym.load_asset(self.sim, asset_root, shelf_asset_file, shelf_asset_options )

    # override franka start pose
    franka_start_pose = gymapi.Transform()
    table_thickness = 0.054
    table_stand_height = 0.01
    franka_start_pose.p = gymapi.Vec3(-0.65, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
    franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # define start pose for box (will be reset later)
    box_height = .04
    box_start_pose = gymapi.Transform()
    box_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+box_height/2)
    box_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # define start pose for shelf (will be reset later)
    shelf_height = 0
    shelf_start_pose = gymapi.Transform()
    shelf_start_pose.p = gymapi.Vec3(.25, 0, self._table_surface_pos[2]+shelf_height/2)
    shelf_start_pose.r = gymapi.Quat( 0, 0, -0.7071068, 0.7071068  )

    texture_handle = self.gym.create_texture_from_file(self.sim, os.path.join(asset_root, shelf_texture_path))
    
    
    # ---------------------- Compute aggregate size ----------------------
    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    
    num_object_bodies = self.gym.get_asset_rigid_body_count(box_asset) + self.gym.get_asset_rigid_body_count(shelf_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(box_asset) + self.gym.get_asset_rigid_shape_count(shelf_asset)
    num_object_dofs = self.gym.get_asset_dof_count(box_asset) + self.gym.get_asset_dof_count(shelf_asset)
    
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    print("num object bodies: ", num_object_bodies)
    print("num object dofs: ", num_object_dofs)
    
    
    # ---------------------- Define goals ----------------------
    # obj is used by box, goal is used by shelf
    obj_low =   (-.05, -.10, self._table_surface_pos[2]+box_height/2)
    obj_high =  (-.05,  .10, self._table_surface_pos[2]+box_height/2)

    goal_low =  (.20, -.10, self._table_surface_pos[2] + .3) 
    goal_high = (.30,  .10, self._table_surface_pos[2] + .3)

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
        
        # Create box
        box_actor = self.gym.create_actor(env_ptr, box_asset, box_start_pose, "obj", i, -1, 0)
        self.gym.set_rigid_body_color(
                env_ptr, box_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(.5, .5, .5))

        # create shelf
        shelf_actor = self.gym.create_actor(env_ptr, shelf_asset, shelf_start_pose, "shelf", i, -1, 0)
        self.gym.set_rigid_body_texture(
            env_ptr, shelf_actor, 1, gymapi.MESH_VISUAL, texture_handle)
        
        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                    i, 1, 0)

        # if self.aggregate_mode > 0:
        #     self.gym.end_aggregate(env_ptr)

        envs.append(env_ptr)
        frankas.append(franka_actor)
        objects.append(box_actor)

    # pass these to the main create envs fns
    num_task_actor = 2 # shelf, cylinder
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    return envs, frankas, objects, random_reset_space, num_task_actor, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env
    
    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    cylinder_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    cylinder_rot = self.root_state_tensor[multi_env_ids_int32,3:7]

    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    shelf_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    shelf_rot = self.root_state_tensor[multi_env_ids_int32,3:7]
    
    return torch.cat([
        cylinder_pos,
        cylinder_rot,
        shelf_pos,
        shelf_rot,
    ], dim=-1)


@torch.jit.script
def compute_reward(
    reset_buf, progress_buf, actions, franka_dof_pos, franka_lfinger_pos, franka_rfinger_pos, max_episode_length,
    tcp_init, target_pos, obj, obj_init_pos, specialized_kwargs
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Dict[str,Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    _TARGET_RADIUS = 0.05 
    TABLE_Z = 1.0270

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2

    obj_to_target = torch.norm(obj - target_pos,dim=-1)
    tcp_to_obj = torch.norm(obj - tcp,dim=-1)
    in_place_margin = torch.norm(obj_init_pos-target_pos,dim=-1)

    in_place = tolerance(
        obj_to_target,
        bounds=(0.0, _TARGET_RADIUS),
        margin=in_place_margin,
        sigmoid="long_tail",
    )
    
    # create a normalized 0 to 1 measurement of how open the gripper is, where 1 is fully open and 0 is fully closed
    gripper_distance_apart = torch.norm(franka_rfinger_pos-franka_lfinger_pos,dim=-1)
    tcp_opened = torch.clip(gripper_distance_apart/.095,0.0,1.0)

    object_grasped = _gripper_caging_reward(
        obj,
        franka_lfinger_pos,
        franka_rfinger_pos,
        tcp,
        tcp_init,
        actions,
        obj_init_pos,
        object_reach_radius=0.01,
        obj_radius=0.015,
        pad_success_thresh=0.05,
        xz_thresh=0.005,
        medium_density=True
    )
    
    rewards = hamacher_product(object_grasped,in_place)
    z_offset = TABLE_Z + .3

    z_scaling = torch.clip((z_offset - obj[:,2]) / z_offset,0,1)
    y_scaling = torch.clip((obj[:,0] - (target_pos[:,0] - 3 * _TARGET_RADIUS)) / (3*_TARGET_RADIUS),0,1)

    bound_loss = hamacher_product(y_scaling,z_scaling)
    
    in_place2 = torch.where((TABLE_Z <= obj[:,2]) & (obj[:,2] < z_offset) & 
                            ((target_pos[:,1]-.15) < obj[:,1]) & (obj[:,1] < (target_pos[:,1]+.15)) &
                            ((target_pos[:,0] - 3 * _TARGET_RADIUS) < obj[:,0]) & (obj[:,0] < target_pos[:,0]), 
                                torch.clip(in_place-bound_loss,0.0,1.0), in_place)

    in_place3 = torch.where((TABLE_Z <= obj[:,2]) & (obj[:,2] < z_offset) & 
                            ((target_pos[:,1]-.15) < obj[:,1]) & (obj[:,1] < (target_pos[:,1]+.15)) &
                            (obj[:,0] >= target_pos[:,0]), 
                                0, in_place2)
    
    rewards = torch.where((tcp_to_obj < .025) & (tcp_opened > 0) & ((obj[:,2]-.01) > obj_init_pos[:,2]), rewards + 1.0 + 5.0 * in_place3, rewards)*.01
    
    success = obj_to_target < _TARGET_RADIUS
    rewards = torch.where(success, 10, rewards)

    # Compute resets
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
    cylinder_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    self.root_state_tensor[cylinder_multi_env_ids_int32,:3] = self.obj_init_pos[env_ids]
    self.root_state_tensor[cylinder_multi_env_ids_int32,3:7] = to_torch([0,0,0,1],device=self.device)
    self.root_state_tensor[cylinder_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    shelf_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    self.root_state_tensor[shelf_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,3:].clone() + to_torch([0, 0, -.3],device=self.device)
    self.root_state_tensor[shelf_multi_env_ids_int32,3:7] = to_torch([0, 0, -0.7071068, 0.7071068],device=self.device)

    actor_multi_env_ids_int32 = torch.cat((cylinder_multi_env_ids_int32,shelf_multi_env_ids_int32),dim=-1)

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return self.franka_actor_idx[env_ids], actor_multi_env_ids_int32

