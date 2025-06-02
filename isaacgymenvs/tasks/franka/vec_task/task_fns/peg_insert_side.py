import os
from typing import Dict, Tuple

import torch
import numpy as np
from gym import spaces

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch
from isaacgymenvs.tasks.reward_utils import _gripper_caging_reward, tolerance, hamacher_product, rect_prism_tolerance
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
    peg_asset_file = "assets_v2/unified_objects/peg.xml"
    peg_block_file = "assets_v2/unified_objects/peg_block.xml"

    # create peg asset
    peg_asset_options = gymapi.AssetOptions()
    peg_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    peg_asset_options.fix_base_link = False        
    peg_asset_options.disable_gravity = False
    peg_asset = self.gym.load_asset(self.sim, asset_root, peg_asset_file, peg_asset_options )

    # create peg block asset
    peg_block_asset_options = gymapi.AssetOptions()
    peg_block_asset_options.fix_base_link = True
    peg_block_asset_options.disable_gravity = False
    peg_block_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    peg_block_asset = self.gym.load_asset(self.sim, asset_root, peg_block_file, peg_block_asset_options)

    # define start pose for peg (will be reset later)
    peg_height = 0.03
    peg_start_pose = gymapi.Transform()
    peg_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+peg_height/2)
    peg_start_pose.r = gymapi.Quat(0, 0, 0, 1)

    # define start pose for peg block (will be reset later)
    peg_block_height = 0
    peg_block_start_pose = gymapi.Transform()
    peg_block_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+peg_block_height/2)
    peg_block_start_pose.r = gymapi.Quat(0, 0, 0, 1)
    
    
    # ---------------------- Compute aggregate size ----------------------
    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)

    self.num_peg_insert_side_peg_bodies = self.gym.get_asset_rigid_body_count(peg_asset)
    
    num_object_bodies = self.gym.get_asset_rigid_body_count(peg_asset)  + self.gym.get_asset_rigid_body_count(peg_block_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(peg_asset) + self.gym.get_asset_rigid_shape_count(peg_block_asset)
    num_object_dofs = self.gym.get_asset_dof_count(peg_asset) + self.gym.get_asset_dof_count(peg_block_asset)
    
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    print("num object bodies: ", num_object_bodies)
    print("num object dofs: ", num_object_dofs)
    
    
    # ---------------------- Define goals ----------------------
    # define goals (peg uses obj and peg_block uses goal), add .01 to account for pegGrasp
    obj_low = (-.1, -.2, self._table_surface_pos[2]+peg_height/2+.01)
    obj_high = (.1, 0, self._table_surface_pos[2]+peg_height/2+.01)

    goal_low =  (-.2, .25, self._table_surface_pos[2]+peg_block_height/2)
    goal_high = ( .1, .35, self._table_surface_pos[2]+peg_block_height/2)

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
        
        # Create peg
        peg_actor = self.gym.create_actor(env_ptr, peg_asset, peg_start_pose, "peg", i, -1, 0)
        self.gym.set_rigid_body_color(env_ptr, peg_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 1, 0.3))

        # create peg block
        peg_block_actor = self.gym.create_actor(env_ptr, peg_block_asset, peg_block_start_pose, "peg_block", i, -1, 0)
        
        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                    i, 1, 0)

        # if self.aggregate_mode > 0:
        #     self.gym.end_aggregate(env_ptr)

        envs.append(env_ptr)
        frankas.append(franka_actor)
        objects.append(peg_actor)

    # pass these to the main create envs fns
    num_task_actor = 2  # peg and peg block
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    return envs, frankas, objects, random_reset_space, num_task_actor, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env
    
    # pegGrasp is offset from last franka rigid body
    peg_grasp_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 3
    peg_grasp_rigid_body_states = self.rigid_body_states[peg_grasp_rigid_body_idx].view(-1, 13)

    obj_pos = peg_grasp_rigid_body_states[:, 0:3].clone()
    obj_pos[:,1] -= .02
    obj_quat = peg_grasp_rigid_body_states[:, 3:7]

    # compute peg_head_pos and peg_head_pos_init
    peg_head_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 1
    if self.specialized_kwargs['peg_insert_side'][env_ids[0].item()]['peg_head_pos_init'] is None:
        self.specialized_kwargs['peg_insert_side'][env_ids[0].item()]['peg_head_pos_init'] = self.rigid_body_states[peg_head_rigid_body_idx].view(-1, 13)[:,:3]
    self.specialized_kwargs['peg_insert_side'][env_ids[0].item()]['peg_head_pos'] = self.rigid_body_states[peg_head_rigid_body_idx].view(-1, 13)[:,:3]

    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    peg_block_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    peg_block_quat = self.root_state_tensor[multi_env_ids_int32,3:7]

    # compute four corners of peg block hole
    brc_col_box_1_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + self.num_peg_insert_side_peg_bodies + 3
    tlc_col_box_1_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + self.num_peg_insert_side_peg_bodies + 4
    brc_col_box_2_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + self.num_peg_insert_side_peg_bodies + 5
    tlc_col_box_2_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + self.num_peg_insert_side_peg_bodies + 6
    self.specialized_kwargs['peg_insert_side'][env_ids[0].item()]['brc_col_box_1_pos'] = self.rigid_body_states[brc_col_box_1_rigid_body_idx].view(-1, 13)[:,:3]
    self.specialized_kwargs['peg_insert_side'][env_ids[0].item()]['tlc_col_box_1_pos'] = self.rigid_body_states[tlc_col_box_1_rigid_body_idx].view(-1, 13)[:,:3]
    self.specialized_kwargs['peg_insert_side'][env_ids[0].item()]['brc_col_box_2_pos'] = self.rigid_body_states[brc_col_box_2_rigid_body_idx].view(-1, 13)[:,:3]
    self.specialized_kwargs['peg_insert_side'][env_ids[0].item()]['tlc_col_box_2_pos'] = self.rigid_body_states[tlc_col_box_2_rigid_body_idx].view(-1, 13)[:,:3]

    return torch.cat([
        obj_pos,
        obj_quat,
        peg_block_pos,
        peg_block_quat
    ], dim=-1)

@torch.jit.script
def compute_reward(
        reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, franka_dof_pos: torch.Tensor,
        franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, max_episode_length: float, 
        init_tcp: torch.Tensor, target_pos: torch.Tensor, obj_pos: torch.Tensor, obj_init_pos: torch.Tensor, specialized_kwargs: Dict[str,torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    TARGET_RADIUS = 0.05
    tlc_col_box_1 = specialized_kwargs['tlc_col_box_1_pos']
    brc_col_box_1 = specialized_kwargs['brc_col_box_1_pos']
    tlc_col_box_2 = specialized_kwargs['tlc_col_box_2_pos']
    brc_col_box_2 = specialized_kwargs['brc_col_box_2_pos']

    obj_head = specialized_kwargs['peg_head_pos']
    peg_head_pos_init = specialized_kwargs['peg_head_pos_init']
    scale = specialized_kwargs["scale"]  

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2
    tcp_to_obj = torch.norm(obj_pos - tcp,dim=-1)
    obj_to_target = torch.norm((obj_head - target_pos) * scale,dim=-1)

    in_place_margin = torch.norm((peg_head_pos_init - target_pos)* scale, dim=-1)
    in_place = tolerance(
        obj_to_target,
        bounds=(0.0, TARGET_RADIUS),
        margin=in_place_margin,
        sigmoid="long_tail",
    )

    collision_box_bottom_1 = rect_prism_tolerance(
        curr=obj_head, one=tlc_col_box_1, zero=brc_col_box_1
    )
    collision_box_bottom_2 = rect_prism_tolerance(
        curr=obj_head, one=tlc_col_box_2, zero=brc_col_box_2
    )
    collision_boxes = hamacher_product(
        collision_box_bottom_2, collision_box_bottom_1
    )
    in_place = hamacher_product(in_place, collision_boxes)

    pad_success_margin = 0.03
    object_reach_radius = 0.01
    x_z_margin = 0.005
    obj_radius = 0.0075

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

    # create a normalized 0 to 1 measurement of how open the gripper is, where 1 is fully open and 0 is fully closed
    gripper_distance_apart = torch.norm(franka_rfinger_pos-franka_lfinger_pos,dim=-1)
    tcp_opened = torch.clip(gripper_distance_apart/.095,0.0,1.0)

    object_grasped = torch.where((tcp_to_obj < 0.08) & (tcp_opened > 0) & ((obj_pos[:,2] - .01) > obj_init_pos[:,2]), 1.0, object_grasped)

    in_place_and_object_grasped = hamacher_product(
        object_grasped, in_place
    )

    rewards = in_place_and_object_grasped
    rewards = torch.where((tcp_to_obj < 0.08) & (tcp_opened > 0) & ((obj_pos[:,2] - .01) > obj_init_pos[:,2]), rewards + 1.0 + 5 * in_place, rewards) * .01

    success = obj_to_target <= TARGET_RADIUS
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

    self.obj_init_pos[env_ids] =  self.last_rand_vecs[env_ids,:3]

    # target is only a little bit into the peg block hole 
    # this is (differen than metaworld which offsets by less, (-.03) in y, to get a deeper insertion)
    self.target_pos[env_ids] = self.last_rand_vecs[env_ids,3:].clone() + to_torch([0, -.09, .13],device=self.device)       

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
    peg_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    self.root_state_tensor[peg_multi_env_ids_int32,:3] = self.obj_init_pos[env_ids]
    self.root_state_tensor[peg_multi_env_ids_int32,3:7] = to_torch([0, 0, -0.7071068, 0.7071068],device=self.device)
    self.root_state_tensor[peg_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    peg_block_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    self.root_state_tensor[peg_block_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,3:]
    self.root_state_tensor[peg_block_multi_env_ids_int32,3:7] = to_torch([0,0,0,1],device=self.device)

    actor_multi_env_ids_int32 = torch.cat((peg_multi_env_ids_int32,peg_block_multi_env_ids_int32),dim=-1).flatten().to(dtype=torch.int32)

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return self.franka_actor_idx[env_ids],actor_multi_env_ids_int32