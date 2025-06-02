import os
from typing import Dict, Tuple

import math
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
    # ---------------------- Load assets ----------------------
    self = env
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)

    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../assets")
    door_asset_file = "assets_v2/unified_objects/doorlockB.xml"

    # load door asset
    door_asset_options = gymapi.AssetOptions()
    door_asset_options.fix_base_link = True
    door_asset_options.collapse_fixed_joints = False
    door_asset_options.disable_gravity = False
    door_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    door_asset_options.replace_cylinder_with_capsule = False
    door_asset = self.gym.load_asset(self.sim, asset_root, door_asset_file, door_asset_options)

    # set door dof properties
    door_dof_props = self.gym.get_asset_dof_properties(door_asset)
    num_door_dofs = self.gym.get_asset_dof_count(door_asset)
    
    
    door_dof_lower_limits = []
    door_dof_upper_limits = []
    for i in range(num_door_dofs):
        door_dof_lower_limits.append(door_dof_props['lower'][i])
        door_dof_upper_limits.append(door_dof_props['upper'][i])
        # door_dof_props['damping'][i] = 10.0
    
    self.door_dof_lower_limits = to_torch(door_dof_lower_limits, device=self.device)
    self.door_dof_upper_limits = to_torch(door_dof_upper_limits, device=self.device)

    # Define start pose for door (going to be reset anyway)
    door_height = .3
    door_start_pose = gymapi.Transform()
    door_start_pose.p = gymapi.Vec3(.3,0,self._table_surface_pos[2]+door_height/2)
    door_start_pose.r = gymapi.Quat( 0, 0, -.7, .7)


    # ---------------------- Compute aggregate size ----------------------
    num_object_bodies = self.gym.get_asset_rigid_body_count(door_asset)
    num_object_dofs = self.gym.get_asset_dof_count(door_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(door_asset)

    print("num door bodies: ", num_object_bodies)
    print("num door dofs: ", num_object_dofs)

    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2


    # ---------------------- Define goals ----------------------    
    # obj is used by door
    obj_low =  (.35, -.1, self._table_surface_pos[2]+door_height/2)
    obj_high = (.45, 0.0, self._table_surface_pos[2]+door_height/2)
    goal_low = (0, 0, 0)
    goal_high = (0, 0, 0)
    
    # check whether they are the same accross the tasks
    table_pos = [0.0, 0.0, 1.0]
    table_thickness = 0.054

    goal_space = spaces.Box(np.array(goal_low),np.array(goal_high))
    random_reset_space = spaces.Box(
        np.hstack((obj_low, goal_low)),
        np.hstack((obj_high, goal_high)),
    )


    # ---------------------- Create envs ----------------------
    frankas = []
    envs = []
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

        door_pose = door_start_pose
        door_actor = self.gym.create_actor(env_ptr, door_asset, door_pose, "door", i, -1, 0)
        self.gym.set_actor_dof_properties(env_ptr, door_actor, door_dof_props)

        if self.aggregate_mode == 1:
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
        
        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                    i, 1, 0)

        # if self.aggregate_mode > 0:
        #     self.gym.end_aggregate(env_ptr)

        envs.append(env_ptr)
        frankas.append(franka_actor)
        objects.append(door_actor)
        
    # pass these to the main create envs fns
    num_task_actors = 1  # this sence only has 1 actor except for franka and table
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    # print(self.gym.get_actor_rigid_body_dict(env_ptr, door_actor))
    return envs, frankas, objects, random_reset_space, num_task_actors, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env

    # handle rigid body idx is offset from last franka rigid body
    door_handle_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 4
    door_rigid_body_states = self.rigid_body_states[door_handle_rigid_body_idx].view(-1, 13)
    door_pos = door_rigid_body_states[:, 0:3]
    door_rot = door_rigid_body_states[:, 3:7]

    # door dof pos idx is right after last franka dof
    door_dof_pos_idx = self.franka_dof_start_idx[env_ids] + self.num_franka_dofs
    door_dof_pos = self.dof_state[door_dof_pos_idx, 0]

    self.specialized_kwargs['door_close']['door_dof_pos'] = door_dof_pos
  
    return torch.cat([
        door_pos,
        door_rot,
        # this scene only has one object, set the other one to be zero
        torch.zeros_like(door_pos),
        torch.zeros_like(door_rot),
    ], dim=-1)


@torch.jit.script
def compute_reward(
    reset_buf, progress_buf, actions, franka_dof_pos, franka_lfinger_pos, franka_rfinger_pos, max_episode_length,
    tcp_init, target_pos, door_pos, obj_init_pos, specialized_kwargs
):
     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Dict[str,Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    door_dof_pos = specialized_kwargs["door_dof_pos"]

    _TARGET_RADIUS = .05
    tcp = (franka_lfinger_pos + franka_rfinger_pos)/2
    obj = door_pos
    
    tcp_to_target = torch.norm(tcp - target_pos, dim=-1)
    obj_to_target = torch.norm(obj - target_pos, dim=-1)

    in_place_margin = torch.norm(obj - obj_init_pos, dim=-1)
    in_place = tolerance(
        obj_to_target,
        bounds=(0.0, _TARGET_RADIUS),
        margin=in_place_margin,
        sigmoid='gaussian',
    )

    hand_margin = torch.norm(tcp_init - obj, dim=-1)
    hand_in_place = tolerance(
        tcp_to_target,
        bounds=(0.0, _TARGET_RADIUS*.25),
        margin=hand_margin,
        sigmoid='gaussian',
    )

    rewards = (3 * hand_in_place + 6 * in_place) * .01

    success = obj_to_target < _TARGET_RADIUS
    rewards = torch.where(success, 10, rewards)
    
    # reset if success or max length reached
    reset_buf = torch.logical_or(success, progress_buf >= max_episode_length - 1).to(reset_buf.dtype)
    return rewards, reset_buf, success


def reset_env(env,tid, env_ids, random_reset_space):
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

    # obj init pos is the part of the handle connecting to the door when door is partially open
    self.obj_init_pos[env_ids] = self.last_rand_vecs[env_ids,:3] + to_torch([-.425, .25, 0], device=self.device)

    # target pos is root_body + offset, to the position the handle would be when the door is closed
    self.target_pos[env_ids] = self.last_rand_vecs[env_ids,:3] + to_torch([-.16, -.15, 0.0],device=self.device) # metaworld uses -.2,-.2

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
    door_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    self.root_state_tensor[door_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,:3]
    
    # reset door to partially OPEN 
    door_dof_idxs_long = (self.franka_dof_start_idx[env_ids]+self.num_franka_dofs).flatten().to(dtype=torch.long)

    # _dof_state = self.dof_state[door_dof_idxs_long,:].view(-1, 2)
    all_pos = torch.zeros_like(self.dof_state)
    all_pos[...,0][door_dof_idxs_long] = -1.5708
    all_pos[...,1][door_dof_idxs_long] = 0.0

    self.dof_state[...,0].flatten().index_copy_(0,door_dof_idxs_long.flatten(),all_pos[...,0][door_dof_idxs_long].flatten())
    self.dof_state[...,1].flatten().index_copy_(0,door_dof_idxs_long.flatten(),all_pos[...,1][door_dof_idxs_long].flatten())

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return torch.cat((self.franka_actor_idx[env_ids],self.franka_actor_idx[env_ids]+1),dim=-1), door_multi_env_ids_int32