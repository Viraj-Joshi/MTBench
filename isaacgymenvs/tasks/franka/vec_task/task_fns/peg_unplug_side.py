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
    plug_asset_file = "assets_v2/unified_objects/plug.xml"
    plug_wall_asset_file = "assets_v2/unified_objects/plug_wall.xml"

    # load plug asset
    plug_asset_options = gymapi.AssetOptions()
    plug_asset_options.fix_base_link = False
    plug_asset_options.disable_gravity = False
    plug_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    plug_asset_options.replace_cylinder_with_capsule = True
    plug_asset = self.gym.load_asset(self.sim, asset_root, plug_asset_file, plug_asset_options)

    # load plug_wall asset
    plug_wall_asset_options = gymapi.AssetOptions()
    plug_wall_asset_options.fix_base_link = True
    plug_wall_asset_options.disable_gravity = False
    plug_wall_asset = self.gym.load_asset(self.sim, asset_root, plug_wall_asset_file, plug_wall_asset_options )

    # Define start pose for plug (going to be reset anyway)
    plug_height = 0
    plug_start_pose = gymapi.Transform()
    plug_start_pose.p = gymapi.Vec3(.22,-.3,self._table_surface_pos[2]+plug_height/2)
    plug_start_pose.r = gymapi.Quat( 0, 0, 0, 1)

    # define start pose for plug_wall (will NOT be reset later)
    plug_wall_height = 0
    plug_wall_start_pose = gymapi.Transform()
    plug_wall_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+plug_wall_height/2)
    plug_wall_start_pose.r = gymapi.Quat(0, 0.0, 0.0, 1.0)
    
    
    # ---------------------- Compute aggregate size ----------------------
    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    
    num_object_bodies = self.gym.get_asset_rigid_body_count(plug_asset) + self.gym.get_asset_rigid_body_count(plug_wall_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(plug_asset) + self.gym.get_asset_rigid_shape_count(plug_wall_asset)
    num_object_dofs = self.gym.get_asset_dof_count(plug_asset) + self.gym.get_asset_dof_count(plug_wall_asset)
    
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    print("num object bodies: ", num_object_bodies)
    print("num object dofs: ", num_object_dofs)
    
    # ---------------------- Define goals ----------------------
    # plug wall uses obj and plug is offset from wall
    obj_low =  (.2, .15, self._table_surface_pos[2])
    obj_high = (.3, .25, self._table_surface_pos[2])

    goal_low = obj_low + np.array([0, -0.194, 0.131])   # NOT USED
    goal_high = obj_high + np.array([0, -0.194, 0.131]) # NOT USED

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
        
        # create plug
        plug_pose = plug_start_pose
        plug_actor = self.gym.create_actor(env_ptr, plug_asset, plug_pose, "plug", i, -1, 0)
        self.gym.set_rigid_body_color(env_ptr, plug_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 1, 0.3))

        # create plug goal
        plug_wall_actor = self.gym.create_actor(env_ptr, plug_wall_asset, plug_wall_start_pose, "plug_wall", i, -1, 0)
        
        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                    i, 1, 0)

        # if self.aggregate_mode > 0:
        #     self.gym.end_aggregate(env_ptr)

        envs.append(env_ptr)
        frankas.append(franka_actor)
        objects.append(plug_actor)

    # pass these to the main create envs fns
    num_task_actor = 2  # plug and plug wall
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    return envs, frankas, objects, random_reset_space, num_task_actor, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env

    plug_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 1
    plug_rigid_body_states = self.rigid_body_states[plug_rigid_body_idx].view(-1, 13)
    obj_pos = plug_rigid_body_states[:, 0:3]
    obj_rot = plug_rigid_body_states[:, 3:7]

    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    plug_wall_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    plug_wall_rot = self.root_state_tensor[multi_env_ids_int32,3:7]

    # print('target pos: ', self.target_pos[env_ids][0])
    # print('obj pos: ', obj_pos[0])
    
    return torch.cat([
        obj_pos,
        obj_rot,
        plug_wall_pos,
        plug_wall_rot
    ], dim=-1)


@torch.jit.script
def compute_reward(
    reset_buf, progress_buf, actions, franka_dof_pos, franka_lfinger_pos, franka_rfinger_pos, max_episode_length,
    init_tcp, target_pos, obj_pos, obj_init_pos, specialized_kwargs
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Dict[str,Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    TARGET_RADIUS = 0.05    

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2
    obj_to_target  = torch.norm(obj_pos - target_pos,dim=-1)
    in_place_margin = torch.norm(obj_init_pos - target_pos,dim=-1)

    in_place = tolerance(
        obj_to_target,
        bounds=(0.0, TARGET_RADIUS),
        margin=in_place_margin,
        sigmoid="long_tail",
    )
    
    finger1_dof = franka_dof_pos[:,-2]
    finger2_dof = -franka_dof_pos[:,-1]
    # create a normalized 0 to 1 measurement of how open the gripper is, where 1 is fully open and 0 is fully closed
    tcp_opened = (finger1_dof - finger2_dof) / .08
    
    grip_pos = obj_pos.clone()
    grip_pos[:,0] -= .06

    tcp_to_obj = torch.norm(grip_pos - tcp,dim=-1)

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
    
    in_place_and_object_grasped = hamacher_product(
        object_grasped, in_place
    )

    rewards = in_place_and_object_grasped
    rewards = torch.where((tcp_to_obj < .02) & (tcp_opened > 0 ) & ((obj_pos[:,2]-.01) > obj_init_pos[:,2]), rewards + 1.0 + 5.0 * in_place, rewards)*.01

    success = obj_to_target < TARGET_RADIUS
    rewards = torch.where(success, 10, rewards)

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | success, torch.ones_like(reset_buf), reset_buf)

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

    # headEnd in the plug xml is object init position
    self.obj_init_pos[env_ids,:3] = self.last_rand_vecs[env_ids,:3] + to_torch([-.11,0,0.024],device=self.device)

    # hole in the plug_wall xml is the target position
    self.target_pos[env_ids] = self.last_rand_vecs[env_ids,:3] + to_torch([-.02,0,0.131],device=self.device)

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
    self.root_state_tensor[obj_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,:3] + to_torch([-.15,0,0],device=self.device)
    self.root_state_tensor[obj_multi_env_ids_int32,3:7] = to_torch([0,1,0,0],device=self.device)
    self.root_state_tensor[obj_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    plug_wall_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    self.root_state_tensor[plug_wall_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,:3]
    self.root_state_tensor[plug_wall_multi_env_ids_int32,3:7] = to_torch([0,0,1,0],device=self.device)

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    actor_multi_env_ids_int32 = torch.cat((obj_multi_env_ids_int32,plug_wall_multi_env_ids_int32),dim=-1).flatten().to(dtype=torch.int32)

    return self.franka_actor_idx[env_ids], actor_multi_env_ids_int32

