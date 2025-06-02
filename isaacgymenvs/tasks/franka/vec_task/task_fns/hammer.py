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
    # ---------------------- Load assets ----------------------
    self = env
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)

    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../assets")
    hammer_block_asset_file = "assets_v2/unified_objects/hammer_block.xml"
    hammer_asset_file = "assets_v2/unified_objects/hammer.xml"

    # load hammer_block asset
    hammer_block_asset_options = gymapi.AssetOptions()
    hammer_block_asset_options.fix_base_link = True
    hammer_block_asset_options.disable_gravity = False
    hammer_block_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    hammer_block_asset_options.replace_cylinder_with_capsule = True
    hammer_block_asset = self.gym.load_asset(self.sim, asset_root, hammer_block_asset_file, hammer_block_asset_options)
    
    # load hammer asset
    hammer_asset_options = gymapi.AssetOptions()
    hammer_asset_options.fix_base_link = False
    hammer_asset_options.disable_gravity = False
    hammer_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    hammer_asset_options.replace_cylinder_with_capsule = True
    hammer_asset = self.gym.load_asset(self.sim, asset_root, hammer_asset_file, hammer_asset_options)

    # set hammer block dof properties
    hammer_block_dof_props = self.gym.get_asset_dof_properties(hammer_block_asset)
    num_hammer_block_dofs = self.gym.get_asset_dof_count(hammer_block_asset)
    
    self.hammer_block_dof_lower_limits = []
    self.hammer_block_dof_upper_limits = []
    for i in range(num_hammer_block_dofs):
        self.hammer_block_dof_lower_limits.append(hammer_block_dof_props['lower'][i])
        self.hammer_block_dof_upper_limits.append(hammer_block_dof_props['upper'][i])
        # hammer_block_dof_props['damping'][i] = 10.0
    
    self.hammer_block_dof_lower_limits = to_torch(self.hammer_block_dof_lower_limits, device=self.device)
    self.hammer_block_dof_upper_limits = to_torch(self.hammer_block_dof_upper_limits, device=self.device)

    # Define start pose for hammer_block (deterministic)
    hammer_block_height = 0
    hammer_block_start_pose = gymapi.Transform()
    hammer_block_start_pose.p = gymapi.Vec3(.25,-.24,self._table_surface_pos[2]+hammer_block_height/2)
    hammer_block_start_pose.r = gymapi.Quat( 0, 0, -0.7068252, 0.7073883)

    # Define start pose for hammer (random)
    hammer_height = 0.1
    hammer_start_pose = gymapi.Transform()
    hammer_start_pose.p = gymapi.Vec3(-.2,-.24,self._table_surface_pos[2]+hammer_height/2)
    hammer_start_pose.r = gymapi.Quat( 0, 0, -0.7068252, 0.7073883)


    # ---------------------- Compute aggregate size ----------------------
    num_object_bodies = self.gym.get_asset_rigid_body_count(hammer_block_asset) + self.gym.get_asset_rigid_body_count(hammer_asset)
    num_object_dofs = self.gym.get_asset_dof_count(hammer_block_asset) + self.gym.get_asset_dof_count(hammer_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(hammer_block_asset) + self.gym.get_asset_rigid_shape_count(hammer_asset)

    self.num_hammer_block_rigid_bodies = self.gym.get_asset_rigid_body_count(hammer_block_asset)

    print("num bodies: ", num_object_bodies)
    print("num hammer block dofs: ", num_object_dofs)

    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    # check whether they are the same across the tasks
    table_pos = [0.0, 0.0, 1.0]
    table_thickness = 0.054

    # ---------------------- Define goals ----------------------    
    # obj is used for hammer placement
    goal_low = (0, 0, 0)
    goal_high = (0, 0 ,0)
    obj_low =  (-0.2, -0.1, self._table_surface_pos[2]+hammer_height/2)
    obj_high = (-0.1,  0.1, self._table_surface_pos[2]+hammer_height/2)

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

        hammer_block_pose = hammer_block_start_pose
        hammer_block_actor = self.gym.create_actor(env_ptr, hammer_block_asset, hammer_block_pose, "hammer_block", i, -1, 0)
        self.gym.set_actor_dof_properties(env_ptr, hammer_block_actor, hammer_block_dof_props)

        hammer_actor = self.gym.create_actor(env_ptr, hammer_asset, hammer_start_pose, "hammer", i, -1, 0)

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
        objects.append(hammer_actor)
        
    # pass these to the main create envs fns
    num_task_actors = 2 # hammer and hammer block
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies

    return envs, frankas, objects, random_reset_space, num_task_actors, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env

    # hammer block rigid body idx is offset from last hammer rigid body
    hammer_block_nail_rigid_body_idx = self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 2
    hammer_block_nail_rigid_body_states = self.rigid_body_states[hammer_block_nail_rigid_body_idx].view(-1, 13)
    hammer_block_nail_pos = hammer_block_nail_rigid_body_states[:, 0:3]
    hammer_block_nail_rot = hammer_block_nail_rigid_body_states[:, 3:7]

    # hammer rigid body idx is offset from last franka rigid body
    hammer_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + self.num_hammer_block_rigid_bodies + 1
    hammer_rigid_body_states = self.rigid_body_states[hammer_rigid_body_idx].view(-1, 13)
    obj_pos = hammer_rigid_body_states[:, 0:3]
    obj_rot = hammer_rigid_body_states[:, 3:7]

    # get hammer block dof pos
    hammer_block_dof_idx = self.franka_dof_start_idx[env_ids] + self.num_franka_dofs
    hammer_block_dof_pos = self.dof_state[hammer_block_dof_idx,0].unsqueeze(-1)
    
    self.specialized_kwargs["hammer"][env_ids[0].item()]["hammer_block_dof_pos"] = hammer_block_dof_pos
    self.specialized_kwargs["hammer"][env_ids[0].item()]["hammer_rot"] = obj_rot


    target_pos_rigid_body_idx = self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 3
    target_pos_rigid_body_states = self.rigid_body_states[target_pos_rigid_body_idx].view(-1, 13)
    self.target_pos[env_ids] = target_pos_rigid_body_states[:, 0:3]
      
    return torch.cat([
        obj_pos,
        obj_rot,
        hammer_block_nail_pos,
        hammer_block_nail_rot
    ], dim=-1)

@torch.jit.script
def _reward_quat(hammer_rot:torch.Tensor)->torch.Tensor:
    # Ideal laid-down wrench has quat [0, 0, -0.7068252, 0.7073883]
    # Rather than deal with an angle between quaternions, just approximate:
    error = hammer_rot.clone()
    error[:,2] -= -0.7068252
    error[:,3] -= 0.7073883
    error = torch.norm(error,dim=-1)
    return torch.maximum(1.0 - error / 0.4, torch.zeros_like(error))

@torch.jit.script
def _reward_pos(hammer_head: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
    pos_error = torch.norm(target_pos - hammer_head,dim=-1)

    a = 0.1  # Relative importance of just *trying* to lift the hammer
    b = 0.9  # Relative importance of hitting the nail
    TABLE_Z = 1.0270
    lifted = hammer_head[:,2] > (TABLE_Z + 0.04)  # changed from +.02
    in_place = a * lifted + b * tolerance(
        pos_error,
        bounds=(0.0, 0.02),
        margin=torch.ones_like(pos_error) * 0.2,
        sigmoid="long_tail",
    )

    return in_place

@torch.jit.script
def compute_reward(
    reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, franka_dof_pos: torch.Tensor,
    franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, max_episode_length: float, 
    tcp_init: torch.Tensor, target_pos: torch.Tensor, hammer_pos: torch.Tensor, obj_init_pos: torch.Tensor, specialized_kwargs: Dict[str,torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    HAMMER_HANDLE_LENGTH = .14
    hammer_rot = specialized_kwargs["hammer_rot"]
    hammer_block_dof_pos = specialized_kwargs["hammer_block_dof_pos"]

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2
    # add an offset to get to the hammer head
    hammer_head = hammer_pos.clone()
    hammer_head[:,0] += .07
    hammer_head[:,1] += -.16

    threshold = HAMMER_HANDLE_LENGTH/2.0
    hammer_threshed_y = torch.where(torch.abs(hammer_pos[:,1]-tcp[:,1]) < threshold,tcp[:,1], hammer_pos[:,1])
    hammer_threshed = torch.hstack((hammer_pos[:,0].unsqueeze(-1),hammer_threshed_y.unsqueeze(-1),hammer_pos[:,2].unsqueeze(-1)))

    reward_quat = _reward_quat(hammer_rot)

    reward_grab = _gripper_caging_reward(
        hammer_threshed,
        franka_lfinger_pos,
        franka_rfinger_pos,
        tcp,
        tcp_init,
        actions,
        obj_init_pos,
        object_reach_radius=0.01,
        obj_radius=0.015,
        pad_success_thresh=0.02,
        xz_thresh=0.01,
        medium_density=True
    )

    reward_in_place = _reward_pos(hammer_head, target_pos)
    rewards = (2.0 * reward_grab + 6.0 * reward_in_place) * .01

    success = (hammer_block_dof_pos[:,0] > .09) & (rewards > .05)

    pos_error = torch.norm(target_pos - hammer_head,dim=-1)
    # to prevent reward hacking, only give the success reward if hammer is lifted
    rewards = torch.where(success, 20, rewards) 
    
    # reset if hammer_block is pressed or max length reached
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

    # get obj init pos
    self.obj_init_pos[env_ids] = self.last_rand_vecs[env_ids,:3]

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

    # reset obj pose
    obj_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    self.root_state_tensor[obj_multi_env_ids_int32,:3] = self.obj_init_pos[env_ids]
    self.root_state_tensor[obj_multi_env_ids_int32,3:7] = to_torch([0, 0, -0.7068252, 0.7073883],device=self.device)
    self.root_state_tensor[obj_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    # reset hammer_block to unpressed
    hammer_block_dof_idxs_long = (self.franka_dof_start_idx[env_ids]+self.num_franka_dofs).flatten().to(dtype=torch.long)

    all_pos = torch.zeros_like(self.dof_state)
    all_pos[...,0][hammer_block_dof_idxs_long] = self.hammer_block_dof_lower_limits
    all_pos[...,1][hammer_block_dof_idxs_long] = 0.0

    self.dof_state[...,0].flatten().index_copy_(0,hammer_block_dof_idxs_long.flatten(),all_pos[...,0][hammer_block_dof_idxs_long].flatten())
    self.dof_state[...,1].flatten().index_copy_(0,hammer_block_dof_idxs_long.flatten(),all_pos[...,1][hammer_block_dof_idxs_long].flatten())

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    dof_mult_env_ids_int32 = torch.cat((self.franka_actor_idx[env_ids],self.franka_actor_idx[env_ids]+1),dim=-1).flatten().to(dtype=torch.int32)

    return dof_mult_env_ids_int32, obj_multi_env_ids_int32

