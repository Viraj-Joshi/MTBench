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
    peg_asset_file = "assets_v2/unified_objects/assembly_peg.xml"
    round_nut_asset_file = "assets_v2/unified_objects/round_nut.xml"

    # create peg asset
    peg_asset_options = gymapi.AssetOptions()
    peg_asset_options.fix_base_link = True
    peg_asset_options.disable_gravity = False
    peg_asset = self.gym.load_asset(self.sim, asset_root, peg_asset_file, peg_asset_options )

    # create round nut asset
    round_nut_asset_options = gymapi.AssetOptions()
    round_nut_asset_options.fix_base_link = False
    round_nut_asset_options.disable_gravity = False
    round_nut_asset = self.gym.load_asset(self.sim, asset_root, round_nut_asset_file, round_nut_asset_options)

    # define start pose for peg (will be reset later)
    peg_height = .1
    peg_start_pose = gymapi.Transform()
    peg_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+peg_height/2)
    peg_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # define start pose for round nut (will be reset later)
    round_nut_height = 0.04
    round_nut_start_pose = gymapi.Transform()
    round_nut_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+round_nut_height/2)
    round_nut_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    
    # ---------------------- Compute aggregate size ----------------------
    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    
    num_object_bodies = self.gym.get_asset_rigid_body_count(peg_asset)  + self.gym.get_asset_rigid_body_count(round_nut_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(peg_asset) + self.gym.get_asset_rigid_shape_count(round_nut_asset)
    num_object_dofs = self.gym.get_asset_dof_count(peg_asset) + self.gym.get_asset_dof_count(round_nut_asset)
    
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    print("num object bodies: ", num_object_bodies)
    print("num object dofs: ", num_object_dofs)
    
    
    # ---------------------- Define goals ----------------------
    # nut uses obj and peg uses goal
    obj_low = (-.05, 0, self._table_surface_pos[2]+.02)
    obj_high = (-.05, 0, self._table_surface_pos[2]+.02)

    goal_low = (.10, -.1, self._table_surface_pos[2] + .05)
    goal_high = (.20, .1, self._table_surface_pos[2] + .05)

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
        
        # create round nut
        round_nut_actor = self.gym.create_actor(env_ptr, round_nut_asset, round_nut_start_pose, "round_nut", i, -1, 0)

        # Create peg
        peg_actor = self.gym.create_actor(env_ptr, peg_asset, peg_start_pose, "peg", i, -1, 0)
        
        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                    i, 1, 0)

        # if self.aggregate_mode > 0:
        #     self.gym.end_aggregate(env_ptr)

        envs.append(env_ptr)
        frankas.append(franka_actor)
        objects.append(round_nut_asset)

    # pass these to the main create envs fns
    num_task_actor = 2  # peg and nuts
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    return envs, frankas, objects, random_reset_space, num_task_actor, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env
    
    wrench_handle_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 2
    wrench_handle_rigid_body_states = self.rigid_body_states[wrench_handle_rigid_body_idx].view(-1, 13)

    round_nut_center_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 3
    round_nut_center_rigid_body_states = self.rigid_body_states[round_nut_center_rigid_body_idx].view(-1, 13)

    self.specialized_kwargs['assembly']['round_nut_center_pos'] = round_nut_center_rigid_body_states[:, 0:3]
    self.specialized_kwargs['assembly']['round_nut_center_quat'] = round_nut_center_rigid_body_states[:, 3:7]

    round_nut_pos = wrench_handle_rigid_body_states[:, 0:3]
    round_nut_quat = wrench_handle_rigid_body_states[:, 3:7]

    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    peg_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    peg_quat = self.root_state_tensor[multi_env_ids_int32,3:7]

    # print('round_nut center pos',  round_nut_center_rigid_body_states[0, 0:3])
    # print('round_nut_pos', round_nut_pos[0])
    # print('target_pos', self.target_pos[0])
    # print('obj_init_pos', self.obj_init_pos[0])
    
    return torch.cat([
        round_nut_pos,
        round_nut_quat,
        peg_pos,
        peg_quat
    ], dim=-1)


@torch.jit.script
def _reward_quat_assemble(nut_quat:torch.Tensor)->torch.Tensor:
    # Ideal laid-down wrench has quat [0, 0, 0, 1]
    # Rather than deal with an angle between quaternions, just approximate:
    error = nut_quat.clone()
    error[:,-1]-=1
    error = torch.norm(error,dim=-1)
    # Sharpen the penalty significantly to penalize drooping of the heavy end.
    # An error of 0.25 (approx 30 degree rotation) will result in a 0 multiplier.
    return torch.maximum(1.0 - error * 4.0, torch.zeros_like(error))

@torch.jit.script
def _reward_pos_assemble(wrench_center: torch.Tensor, target_pos: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    TABLE_Z = 1.0270
    offset = .01

    pos_error = target_pos - wrench_center
    
    radius = torch.norm(pos_error[:,:2],dim=-1)

    aligned = radius < .02
    hooked = pos_error[:,2] > offset # nut must be at least 3cm below peg top to count as hooked
    success = aligned & hooked
    
    # When far: target is peg top (preserves lift incentive).
    # When XY-close: target drops by hooked margin so in_place peak aligns with success zone.
    threshold = torch.where(success, .02, .01)
    near_peg = radius < 0.05
    target_height = torch.where(near_peg, target_pos[:,2] - offset, target_pos[:,2])

    pos_error[:,2] = target_height - wrench_center[:,2]
    pos_error[:,2] *= 3.0  # Make the z error more important than the xy error
    pos_error = torch.norm(pos_error,dim=-1)

    a = .1  # Relative importance of just *trying* to lift the wrench
    b = .9 # Relative importance of placing the wrench on the peg

    lifted = (wrench_center[:,2] > (TABLE_Z + .02)) | (radius < threshold)
    tol = tolerance(
        pos_error,
        bounds=(0.0, 0.02),
        margin=torch.ones_like(pos_error)*.4,
        sigmoid="long_tail",
    )

    # Explicit radius bonus: clear gradient toward the peg at all distances.
    # Denominator must cover full starting distance (~0.19-0.23) so there's gradient from step 0.
    radius_bonus = 1.0 - torch.clamp(radius / 0.25, min=0.0, max=1.0)

    in_place = a * lifted + b * (0.5 * tol + 0.5 * radius_bonus)

    return in_place, success, radius, lifted, pos_error, target_height, tol, radius_bonus

@torch.jit.script
def compute_reward(
        reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, franka_dof_pos: torch.Tensor,
        franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, max_episode_length: float, 
        init_tcp: torch.Tensor, target_pos: torch.Tensor, wrench_pos: torch.Tensor, obj_init_pos: torch.Tensor, specialized_kwargs: Dict[str,torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2
    wrench_center = specialized_kwargs["round_nut_center_pos"]
    wrench_center_quat = specialized_kwargs["round_nut_center_quat"]

    # `wrench_center` is the round nut (Y=0 in local space).
    # `wrench_pos` is the tip of the handle (Y=-0.13 in local space).
    # Grab the cylinder on the handle at Y=-0.1 (t = 0.1/0.13 ≈ 0.77).
    ideal_grab_point = wrench_center * 0.23 + wrench_pos * 0.77

    reward_grab = _gripper_caging_reward(
        ideal_grab_point,
        franka_lfinger_pos,
        franka_rfinger_pos,
        tcp,
        init_tcp,
        actions,
        obj_init_pos,
        object_reach_radius=0.01,
        obj_radius=0.01,
        pad_success_thresh=0.05,
        xz_thresh=0.005,
        medium_density=True
    )

    # Firm grasp gate: tcp must be close to grab point AND fingers closed around it
    tcp_to_grab = torch.norm(tcp - ideal_grab_point, dim=-1)
    finger_dist = torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim=-1)
    grasped = (tcp_to_grab < 0.02) & (finger_dist < 0.06)

    # Placement reward: guide nut toward the peg (only active when grasped)
    in_place, success, radius, lifted, pos_err, tgt_height, tol, radius_bonus = _reward_pos_assemble(wrench_center, target_pos)
    gated_in_place = torch.where(grasped, in_place, torch.zeros_like(in_place))

    # 2 pts for grasping, 8 pts for placing nut on peg (only if firmly grasped)
    rewards = (2.0 * reward_grab + 8.0 * gated_in_place) * .01

    rewards = torch.where(success, 10.0, rewards)

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

    # get random target
    self.target_pos[env_ids] = self.last_rand_vecs[env_ids,3:] + to_torch([0,0,.05],device=self.device)

    # get random nut positions
    # ideal_grab_point is the cylinder at Y=-0.1 in local space (77% from nut to tip).
    self.obj_init_pos[env_ids] =  self.last_rand_vecs[env_ids,:3] + to_torch([0, -0.1, 0],device=self.device)

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
    peg_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    
    # peg pos is the target pos minus a small offset to match height of peg
    self.root_state_tensor[peg_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,3:] - to_torch([0,0,.05],device=self.device)
    self.root_state_tensor[peg_multi_env_ids_int32,3:7] = to_torch([0,0,0,1],device=self.device)
    self.root_state_tensor[peg_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    nut_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    self.root_state_tensor[nut_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,:3]
    self.root_state_tensor[nut_multi_env_ids_int32,3:7] = to_torch([0,0,0,1],device=self.device)
    self.root_state_tensor[nut_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    actor_multi_env_ids_int32 = torch.cat((peg_multi_env_ids_int32,nut_multi_env_ids_int32),dim=-1).flatten().to(dtype=torch.int32)

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return self.franka_actor_idx[env_ids], actor_multi_env_ids_int32

