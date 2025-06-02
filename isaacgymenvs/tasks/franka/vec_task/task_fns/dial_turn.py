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
    dial_asset_file = "assets_v2/unified_objects/dial.xml"

    # load dial asset
    dial_asset_options = gymapi.AssetOptions()
    dial_asset_options.fix_base_link = True
    dial_asset_options.disable_gravity = False
    dial_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    dial_asset_options.replace_cylinder_with_capsule = False
    dial_asset = self.gym.load_asset(self.sim, asset_root, dial_asset_file, dial_asset_options)

    # set door dof properties
    dial_dof_props = self.gym.get_asset_dof_properties(dial_asset)
    num_dial_dofs = self.gym.get_asset_dof_count(dial_asset)
    
    dial_dof_lower_limits = []
    dial_dof_upper_limits = []
    for i in range(num_dial_dofs):
        dial_dof_lower_limits.append(dial_dof_props['lower'][i])
        dial_dof_upper_limits.append(dial_dof_props['upper'][i])
        # door_dof_props['damping'][i] = 10.0
    
    self.dial_dof_lower_limits = to_torch(dial_dof_lower_limits, device=self.device)
    self.dial_dof_upper_limits = to_torch(dial_dof_upper_limits, device=self.device)

    # Define start pose for dial (going to be reset anyway)
    dial_height = 0
    dial_start_pose = gymapi.Transform()
    dial_start_pose.p = gymapi.Vec3(.3,0,self._table_surface_pos[2]+dial_height/2)
    dial_start_pose.r = gymapi.Quat( 0, 0, -0.7071068, 0.7071068)


    # ---------------------- Compute aggregate size ----------------------
    num_object_bodies = self.gym.get_asset_rigid_body_count(dial_asset)
    num_object_dofs = self.gym.get_asset_dof_count(dial_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(dial_asset)

    print("num bodies: ", num_object_bodies)
    print("num dial dofs: ", num_object_dofs)

    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    # check whether they are the same across the tasks
    table_pos = [0.0, 0.0, 1.0]
    table_thickness = 0.054

    # ---------------------- Define goals ----------------------    
    # obj is used for the dial
    obj_low = (.1, -.1, self._table_surface_pos[2]+dial_height/2)
    obj_high = (.2, .1, self._table_surface_pos[2]+dial_height/2)
    goal_low = (0, 0, 0)
    goal_high = (0, 0 ,0)

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

        dial_pose = dial_start_pose
        dial_actor = self.gym.create_actor(env_ptr, dial_asset, dial_pose, "dial", i, -1, 0)
        self.gym.set_actor_dof_properties(env_ptr, dial_actor, dial_dof_props)

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
        objects.append(dial_actor)
        
    # pass these to the main create envs fns
    num_task_actors = 1  # this sence only has 1 actor except for franka and table
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies

    return envs, frankas, objects, random_reset_space, num_task_actors, num_task_dofs, num_task_bodies

def _get_pos_objects_dial_turn(dial_pos, dial_dof_pos):
        dial_center = dial_pos.clone()
        dial_angle_rad = dial_dof_pos.unsqueeze(-1)

        offset = torch.hstack([-torch.cos(dial_angle_rad), torch.sin(dial_angle_rad), torch.zeros_like(dial_angle_rad)])
        dial_radius = 0.05

        offset *= dial_radius

        return dial_center + offset

def compute_observations(env, env_ids):
    self = env

    dial_pos = self.root_state_tensor[self.franka_actor_idx[env_ids]+1,:3]
    dial_quat = self.root_state_tensor[self.franka_actor_idx[env_ids]+1,3:7]

    dial_dof_pos_idx = self.franka_dof_start_idx[env_ids] + self.num_franka_dofs
    dial_dof_pos = self.dof_state[dial_dof_pos_idx,0]

    dial_push_position = _get_pos_objects_dial_turn(dial_pos, dial_dof_pos)

    if self.specialized_kwargs['dial_turn'][env_ids[0].item()]['dial_push_position_init'] is None:
        self.specialized_kwargs['dial_turn'][env_ids[0].item()]['dial_push_position_init'] = _get_pos_objects_dial_turn(dial_pos, dial_dof_pos)
      
    return torch.cat([
        dial_push_position,
        dial_quat,
        # this scene only has one object, set the other one to be zero
        torch.zeros_like(dial_push_position),
        torch.zeros_like(dial_quat),
    ], dim=-1)


@torch.jit.script
def compute_reward(
    reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, franka_dof_pos: torch.Tensor,
    franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, max_episode_length: float, 
    tcp_init: torch.Tensor, target_pos: torch.Tensor, obj_pos: torch.Tensor, obj_init_pos: torch.Tensor, specialized_kwargs: Dict[str,torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    TARGET_RADIUS = 0.06 # changed from .07
    dial_push_position_init = specialized_kwargs["dial_push_position_init"]

    dial_push_position = obj_pos.clone()
    dial_push_position[:,2] += .09
    dial_push_position[:,1] += -.05
    dial_push_position[:,0] += .02 

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2

    target_to_obj = torch.norm(obj_pos - target_pos,dim=-1)
    target_to_obj_init = torch.norm(dial_push_position_init - target_pos,dim=-1)

    in_place = tolerance(
        target_to_obj,
        bounds=(0.0, TARGET_RADIUS),
        margin=torch.abs(target_to_obj_init - TARGET_RADIUS),
        sigmoid="long_tail",
    )

    dial_reach_radius = 0.005
    tcp_to_obj = torch.norm(dial_push_position - tcp,dim=-1)
    tcp_to_obj_init = torch.norm(dial_push_position_init - tcp_init,dim=-1)

    reach = tolerance(
        tcp_to_obj,
        bounds=(0.0, dial_reach_radius),
        margin=torch.abs(tcp_to_obj_init - dial_reach_radius),
        sigmoid="gaussian",
    )

    gripper_distance_apart = torch.norm(franka_rfinger_pos-franka_lfinger_pos,dim=-1)
    normalized_closedness = 1-torch.clip(gripper_distance_apart/.095,0.0,1.0)

    reach = hamacher_product(reach, normalized_closedness)
    rewards = hamacher_product(reach, in_place) * .01

    # reset if dial turns enough
    success = target_to_obj <= TARGET_RADIUS
    rewards = torch.where(success, 10, rewards)

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

    # the target is just the obj init pos offset by the dial rotating
    self.target_pos[env_ids] = self.obj_init_pos[env_ids].clone() + to_torch([-0.03, 0, 0.03],device=self.device)

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
    dial_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    self.root_state_tensor[dial_multi_env_ids_int32,:3] = self.obj_init_pos[env_ids]

    # reset dial to unturned
    dial_dof_idxs_long = (self.franka_dof_start_idx[env_ids]+self.num_franka_dofs).flatten().to(dtype=torch.long)

    all_pos = torch.zeros_like(self.dof_state)
    all_pos[...,0][dial_dof_idxs_long] = self.dial_dof_lower_limits
    all_pos[...,1][dial_dof_idxs_long] = 0.0

    self.dof_state[...,0].flatten().index_copy_(0,dial_dof_idxs_long.flatten(),all_pos[...,0][dial_dof_idxs_long].flatten())
    self.dof_state[...,1].flatten().index_copy_(0,dial_dof_idxs_long.flatten(),all_pos[...,1][dial_dof_idxs_long].flatten())

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return torch.cat((self.franka_actor_idx[env_ids],self.franka_actor_idx[env_ids]+1),dim=-1), dial_multi_env_ids_int32

