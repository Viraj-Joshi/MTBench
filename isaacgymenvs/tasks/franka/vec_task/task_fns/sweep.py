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
    block_asset_file = "assets_v2/unified_objects/block.xml"
    cylinder_asset_file = "assets_v2/unified_objects/cylinder.xml"

    # create block asset
    block_asset_options = gymapi.AssetOptions()
    block_asset_options.fix_base_link = False
    block_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    block_asset_options.disable_gravity = False
    block_asset = self.gym.load_asset(self.sim, asset_root, block_asset_file, block_asset_options )

    # define start pose for block (will be reset later)
    block_height = 0.04
    block_start_pose = gymapi.Transform()
    block_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+block_height/2)
    block_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    
    # ---------------------- Compute aggregate size ----------------------
    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    
    num_object_bodies = self.gym.get_asset_rigid_body_count(block_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(block_asset)
    num_object_dofs = self.gym.get_asset_dof_count(block_asset)
    
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    print("num object bodies: ", num_object_bodies)
    print("num object dofs: ", num_object_dofs)
    
    
    # ---------------------- Define goals ----------------------
    # obj is for cube and goal is for target
    obj_low = (0, -.1, self._table_surface_pos[2]+block_height/2)
    obj_high = (.1, .1, self._table_surface_pos[2]+block_height/2)

    goal_low = (0, -.4, self._table_surface_pos[2]+block_height/2)
    goal_high = (.1, -.4, self._table_surface_pos[2]+block_height/2)

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
        
        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
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

    if self.specialized_kwargs['sweep']['init_left_pad'] is None:
        self.specialized_kwargs['sweep']['init_left_pad'] = self.franka_lfinger_pos[env_ids]
    if self.specialized_kwargs['sweep']['init_right_pad'] is None:
        self.specialized_kwargs['sweep']['init_right_pad'] = self.franka_rfinger_pos[env_ids]
    
    return torch.cat([
        obj_pos,
        obj_rot,
        # this scence only has one object, set the other one to be zero
        torch.zeros_like(obj_pos),
        torch.zeros_like(obj_rot),
    ], dim=-1)

@torch.jit.script
def _gripper_caging_reward(
    obj_pos : torch.Tensor,
    franka_lfinger_pos : torch.Tensor,
    franka_rfinger_pos : torch.Tensor,
    tcp_center : torch.Tensor,
    init_tcp : torch.Tensor,
    actions : torch.Tensor,
    obj_init_pos : torch.Tensor,
    specialized_kwargs : Dict[str,torch.Tensor],
    obj_radius : float,
    pad_success_thresh : float = 0, # All of these args are unused
    object_reach_radius : float = 0,
    xz_thresh : float = 0,
    desired_gripper_effort : float = 1.0,
    high_density : bool = False,
    medium_density : bool = False
) -> torch.Tensor:
    """Reward for agent grasping obj.

    Args:
        obj_pos,
        franka_lfinger_pos
        franka_rfinger_pos
        tcp_center
        init_tcp
        actions
        obj_init_pos
        obj_radius(float):radius of object's bounding sphere
        specialized_kwargs(Dict[str,torch.Tensor]): specialized kwargs for the task.
        pad_success_thresh(float): successful distance of gripper_pad
            to object
        object_reach_radius(float): successful distance of gripper center
            to the object.
        xz_thresh(float): successful distance of gripper in x_z axis to the
            object. Y axis not included since the caging function handles
                successful grasping in the Y axis.
        desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
        high_density(bool): flag for high-density. Cannot be used with medium-density.
        medium_density(bool): flag for medium-density. Cannot be used with high-density.
    """
    pad_success_margin = 0.05
    grip_success_margin = obj_radius + 0.01
    x_z_success_margin = 0.005

    left_pad = franka_lfinger_pos
    right_pad = franka_rfinger_pos
    init_left_pad = specialized_kwargs["init_left_pad"]
    init_right_pad = specialized_kwargs["init_right_pad"]

    delta_object_y_left_pad = left_pad[:,0] - obj_pos[:,0]
    delta_object_y_right_pad = obj_pos[:,0] - right_pad[:,0]

    right_caging_margin = torch.abs(
        torch.abs(obj_pos[:,0] - init_right_pad[:,0]) - pad_success_margin
    )
    left_caging_margin = torch.abs(
        torch.abs(obj_pos[:,0] - init_left_pad[:,0]) - pad_success_margin
    )
    right_caging = tolerance(
        delta_object_y_right_pad,
        bounds=(obj_radius, pad_success_margin),
        margin=right_caging_margin,
        sigmoid="long_tail",
    )
    left_caging = tolerance(
        delta_object_y_left_pad,
        bounds=(obj_radius, pad_success_margin),
        margin=left_caging_margin,
        sigmoid="long_tail",
    )
    right_gripping = tolerance(
        delta_object_y_right_pad,
        bounds=(obj_radius, grip_success_margin),
        margin=right_caging_margin,
        sigmoid="long_tail",
    )
    left_gripping = tolerance(
        delta_object_y_left_pad,
        bounds=(obj_radius, grip_success_margin),
        margin=left_caging_margin,
        sigmoid="long_tail",
    )

    y_caging = hamacher_product(right_caging, left_caging)
    y_gripping = hamacher_product(right_gripping, left_gripping)

    xz = [1, 2]
    tcp_xz = tcp_center[:,xz]
   
    obj_position_x_z = obj_pos[:,xz]
    tcp_obj_norm_x_z = torch.norm(tcp_xz - obj_position_x_z,dim=-1)

    init_obj_x_z = obj_init_pos[:,xz]
    init_tcp_x_z = init_tcp[:,xz]

    tcp_obj_x_z_margin = (
        torch.maximum(torch.norm(init_obj_x_z - init_tcp_x_z, dim=-1) - x_z_success_margin,torch.zeros_like(init_obj_x_z[:,0]))
    )

    x_z_caging = tolerance(
        tcp_obj_norm_x_z,
        bounds=(0.0, x_z_success_margin),
        margin=tcp_obj_x_z_margin,
        sigmoid="long_tail",
    )

    # Closed-extent gripper information for caging reward-------------
    # gripper_closed = (
    #     torch.minimum(torch.maximum(torch.zeros_like(actions[:,-1]), actions[:,-1]),torch.ones_like(actions[:,-1]) * desired_gripper_effort)/desired_gripper_effort
    # )
    caging = hamacher_product(y_caging, x_z_caging)

    gripping = torch.where(caging > 0.95,y_gripping,0)

    caging_and_gripping = (caging + gripping) / 2

    return caging_and_gripping

from isaacgymenvs.tasks.reward_utils import _gripper_caging_reward as default_gripper_caging_reward

@torch.jit.script
def compute_reward(
    reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, franka_dof_pos: torch.Tensor,
    franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, max_episode_length: float, 
    tcp_init: torch.Tensor, target_pos: torch.Tensor, obj_pos: torch.Tensor, obj_init_pos: torch.Tensor, specialized_kwargs: Dict[str,torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    TARGET_RADIUS = 0.05
    scale = specialized_kwargs["scale"]  

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2
    obj_to_target  = torch.norm((obj_pos - target_pos)*scale,dim=-1)
    tcp_to_obj = torch.norm(obj_pos - tcp,dim=-1)
    in_place_margin = torch.norm(obj_init_pos - target_pos,dim=-1)

    in_place = tolerance(
        obj_to_target,
        bounds=(0.0, TARGET_RADIUS),
        margin=in_place_margin,
        sigmoid="long_tail",
    )

    object_grasped = default_gripper_caging_reward(
        obj_pos,
        franka_lfinger_pos,
        franka_rfinger_pos,
        tcp,
        tcp_init,
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

    rewards = (8 * in_place_and_object_grasped + 2 * in_place)*.01

    success = (obj_to_target < TARGET_RADIUS)
    rewards = torch.where(success, 20, rewards)

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

    self.obj_init_pos[env_ids] = self.last_rand_vecs[env_ids,:3]
    self.target_pos[env_ids] = self.last_rand_vecs[env_ids,3:]
    self.target_pos[env_ids,0] = self.obj_init_pos[env_ids,0]

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
    actor_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    self.root_state_tensor[actor_multi_env_ids_int32,:3] = self.obj_init_pos[env_ids]
    self.root_state_tensor[actor_multi_env_ids_int32,3:7] = to_torch([0,0,0,1],device=self.device)
    self.root_state_tensor[actor_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return self.franka_actor_idx[env_ids], actor_multi_env_ids_int32

