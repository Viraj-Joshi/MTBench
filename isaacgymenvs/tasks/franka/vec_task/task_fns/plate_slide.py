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
    puck_asset_file = "assets_v2/unified_objects/puck.xml"
    puck_goal_asset_file = "assets_v2/unified_objects/puck_goal.xml"

    # load puck asset
    puck_asset_options = gymapi.AssetOptions()
    puck_asset_options.fix_base_link = False
    puck_asset_options.disable_gravity = False
    puck_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    puck_asset_options.replace_cylinder_with_capsule = True
    puck_asset = self.gym.load_asset(self.sim, asset_root, puck_asset_file, puck_asset_options)

    # load puck_goal asset
    puck_goal_asset_options = gymapi.AssetOptions()
    puck_goal_asset_options.fix_base_link = True
    puck_goal_asset_options.disable_gravity = False
    puck_goal_asset = self.gym.load_asset(self.sim, asset_root, puck_goal_asset_file, puck_goal_asset_options )

    # Define start pose for puck (going to be reset anyway)
    puck_height = 0.04
    puck_start_pose = gymapi.Transform()
    puck_start_pose.p = gymapi.Vec3(.22,-.3,self._table_surface_pos[2]+puck_height/2)
    puck_start_pose.r = gymapi.Quat( 0, 0, 0, 1)

    # define start pose for puck_goal (will be reset later)
    puck_goal_height = .03
    puck_goal_start_pose = gymapi.Transform()
    puck_goal_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+puck_goal_height/2)
    puck_goal_start_pose.r = gymapi.Quat(0, 0.0 , -0.7071, 0.7071)
    
    
    # ---------------------- Compute aggregate size ----------------------
    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    
    num_object_bodies = self.gym.get_asset_rigid_body_count(puck_asset) + self.gym.get_asset_rigid_body_count(puck_goal_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(puck_asset) + self.gym.get_asset_rigid_shape_count(puck_goal_asset)
    num_object_dofs = self.gym.get_asset_dof_count(puck_asset) + self.gym.get_asset_dof_count(puck_goal_asset)
    
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    print("num object bodies: ", num_object_bodies)
    print("num object dofs: ", num_object_dofs)
    
    # ---------------------- Define goals ----------------------
    # obj is used for puck_goal placement and goal is the target pos
    obj_low = (0, 0, self._table_surface_pos[2])
    obj_high = (0, 0, self._table_surface_pos[2])
    goal_low = (.25, -.1, self._table_surface_pos[2])
    goal_high = (.3,  .1, self._table_surface_pos[2])

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
        
        # create puck
        puck_pose = puck_start_pose
        puck_actor = self.gym.create_actor(env_ptr, puck_asset, puck_pose, "puck", i, -1, 0)

        # create puck goal
        puck_goal_actor = self.gym.create_actor(env_ptr, puck_goal_asset, puck_goal_start_pose, "puck_goal", i, -1, 0)
        
        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                    i, 1, 0)

        # if self.aggregate_mode > 0:
        #     self.gym.end_aggregate(env_ptr)

        envs.append(env_ptr)
        frankas.append(franka_actor)
        objects.append(puck_actor)

    # pass these to the main create envs fns
    num_task_actor = 2  # puck and puck goal
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    return envs, frankas, objects, random_reset_space, num_task_actor, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env
    
    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    obj_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    obj_rot = self.root_state_tensor[multi_env_ids_int32,3:7]

    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    puck_goal_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    puck_goal_rot = self.root_state_tensor[multi_env_ids_int32,3:7]
    
    return torch.cat([
        obj_pos,
        obj_rot,
        puck_goal_pos,
        puck_goal_rot
    ], dim=-1)


@torch.jit.script
def compute_reward(
    reset_buf, progress_buf, actions, franka_dof_pos, franka_lfinger_pos, franka_rfinger_pos, max_episode_length,
    init_tcp, target_pos, obj, obj_init_pos, specialized_kwargs
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Dict[str,Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    TARGET_RADIUS = .05

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2

    obj_to_target = torch.norm(obj - target_pos,dim=-1)
    in_place_margin = torch.norm(obj_init_pos - target_pos,dim=-1)

    in_place = tolerance(
        obj_to_target,
        bounds = (0.0,TARGET_RADIUS),
        margin = torch.maximum(in_place_margin - TARGET_RADIUS,torch.zeros_like(in_place_margin)),
        sigmoid = 'long_tail'
    )

    tcp_to_obj = torch.norm(tcp-obj,dim=-1)
    obj_grasped_margin = torch.norm(init_tcp - obj_init_pos,dim=-1)

    object_grasped = tolerance(
        tcp_to_obj,
        bounds = (0.0, TARGET_RADIUS),
        margin = torch.maximum(obj_grasped_margin - TARGET_RADIUS,torch.zeros_like(obj_grasped_margin)),
        sigmoid = "long_tail",
    )

    in_place_and_object_grasped = hamacher_product(
        object_grasped, in_place
    )
    rewards = 8 * in_place_and_object_grasped * .01
    
    success = (obj_to_target < TARGET_RADIUS)
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

    # get obj init pos (is the puck)
    self.obj_init_pos[env_ids] = self.last_rand_vecs[env_ids,:3].clone() + to_torch([0, 0, .015],device=self.device)

    # puck goal is last 3 elements of the random vec + offset
    
    # target is offset from puck_goal
    self.target_pos[env_ids] = self.last_rand_vecs[env_ids,3:].clone() + to_torch([-.025,0,0],device=self.device) # added an offset to make it easier

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

    puck_goal_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    self.root_state_tensor[puck_goal_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,3:].clone() + to_torch([0,0,-.015],device=self.device)

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    actor_multi_env_ids_int32 = torch.cat((obj_multi_env_ids_int32,puck_goal_multi_env_ids_int32),dim=-1).flatten().to(dtype=torch.int32)

    return self.franka_actor_idx[env_ids], actor_multi_env_ids_int32

