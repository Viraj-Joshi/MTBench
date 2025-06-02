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
    # ---------------------- Load assets ----------------------
    self = env
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)

    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../assets")
    buttonbox_asset_file = "assets_v2/unified_objects/buttonbox.xml"

    # load buttonbox asset
    buttonbox_asset_options = gymapi.AssetOptions()
    buttonbox_asset_options.fix_base_link = True
    buttonbox_asset_options.disable_gravity = True
    buttonbox_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    buttonbox_asset_options.replace_cylinder_with_capsule = True
    buttonbox_asset = self.gym.load_asset(self.sim, asset_root, buttonbox_asset_file, buttonbox_asset_options)

    # set buttonbox dof properties
    buttonbox_dof_props = self.gym.get_asset_dof_properties(buttonbox_asset)
    num_buttonbox_dofs = self.gym.get_asset_dof_count(buttonbox_asset)
    
    buttonbox_dof_lower_limits = []
    buttonbox_dof_upper_limits = []
    for i in range(num_buttonbox_dofs):
        buttonbox_dof_lower_limits.append(buttonbox_dof_props['lower'][i])
        buttonbox_dof_upper_limits.append(buttonbox_dof_props['upper'][i])
        buttonbox_dof_props['damping'][i] = 10.0
    
    self.buttonbox_dof_lower_limits = to_torch(buttonbox_dof_lower_limits, device=self.device)
    self.buttonbox_dof_upper_limits = to_torch(buttonbox_dof_upper_limits, device=self.device)

    # Define start pose for buttonbox (going to be reset anyway)
    buttonbox_height = .24
    buttonbox_button_pose = gymapi.Transform()
    buttonbox_button_pose.p = gymapi.Vec3(.22,-.3,self._table_surface_pos[2]+buttonbox_height/2)
    buttonbox_button_pose.r = gymapi.Quat(-0.7071, 0, 0, 0.7071)


    # ---------------------- Compute aggregate size ----------------------
    num_object_bodies = self.gym.get_asset_rigid_body_count(buttonbox_asset)
    num_object_dofs = self.gym.get_asset_dof_count(buttonbox_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(buttonbox_asset)

    print("num bodies: ", num_object_bodies)
    print("num buttonbox dofs: ", num_object_dofs)

    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    # check whether they are the same across the tasks
    table_pos = [0.0, 0.0, 1.0]
    table_thickness = 0.054

    # ---------------------- Define goals ----------------------    
    # obj is used for buttonbox placement
    goal_low = (0, 0, 0)
    goal_high = (0, 0, 0)
    obj_low = (.2, -.1, self._table_surface_pos[2]+buttonbox_height/2)
    obj_high = (.3, .1, self._table_surface_pos[2]+buttonbox_height/2)

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

        buttonbox_pose = buttonbox_button_pose
        buttonbox_actor = self.gym.create_actor(env_ptr, buttonbox_asset, buttonbox_pose, "buttonbox", i, -1, 0)
        self.gym.set_actor_dof_properties(env_ptr, buttonbox_actor, buttonbox_dof_props)

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
        objects.append(buttonbox_actor)
        
    # pass these to the main create envs fns
    num_task_actors = 1  # this sence only has 1 actor except for franka and table
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies

    return envs, frankas, objects, random_reset_space, num_task_actors, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env

    # buttonStart rigid body idx is offset from last franka rigid body
    buttonbox_button_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 3
    buttonbox_button_rigid_body_states = self.rigid_body_states[buttonbox_button_rigid_body_idx].view(-1, 13)
    obj_pos = buttonbox_button_rigid_body_states[:, 0:3]
    obj_quat = buttonbox_button_rigid_body_states[:, 3:7]

    return torch.cat([
        obj_pos,
        obj_quat,
        # this scene only has one object, set the other one to be zero
        torch.zeros_like(obj_pos),
        torch.zeros_like(obj_quat),
    ], dim=-1)


@torch.jit.script
def compute_reward(
    reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, franka_dof_pos: torch.Tensor,
    franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, max_episode_length: float, 
    tcp_init: torch.Tensor, target_pos: torch.Tensor, obj_pos: torch.Tensor, obj_init_pos: torch.Tensor, specialized_kwargs: Dict[str,torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2
    obj_pos = obj_pos.clone()
    err = obj_pos - tcp

    err_init = obj_init_pos - tcp_init
    
    tcp_to_obj = torch.norm(err,dim=-1)
    tcp_to_obj_init = torch.norm(err_init,dim=-1)
    obj_to_target = torch.abs(target_pos[:,2]-obj_pos[:,2])
    
    obj_to_target_init = torch.norm(target_pos-obj_init_pos, dim=-1) # changing to abs will make it not learn

    gripper_distance_apart = torch.norm(franka_rfinger_pos-franka_lfinger_pos,dim=-1)
    normalized_openess = torch.clip(gripper_distance_apart/.095,0.0,1.0)
    
    tcp_closed = 1-normalized_openess
    near_button = tolerance(
        tcp_to_obj,
        bounds=(0.0, 0.05),
        margin=tcp_to_obj_init,
        sigmoid="long_tail",
    )

    button_pressed = tolerance(
        obj_to_target,
        bounds=(0.0, 0.005),
        margin=obj_to_target_init,
        sigmoid="long_tail",
    )

    rewards = (2 * hamacher_product(tcp_closed, near_button)) * .01
    success = obj_to_target <= 0.01
    rewards = torch.where(success & (tcp_to_obj < .07), rewards + 16 * button_pressed, rewards)
    # reset if success or max length reached
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | success, torch.ones_like(reset_buf), reset_buf)

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


    # get obj init pos, the position of buttonStart
    self.obj_init_pos[env_ids] = self.last_rand_vecs[env_ids,:3].clone()
    self.obj_init_pos[env_ids,2] += .1935
    
    # the target is just the hole tag
    self.target_pos[env_ids] = self.obj_init_pos[env_ids,:3].clone()
    self.target_pos[env_ids,2] -= .06

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
    buttonbox_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    self.root_state_tensor[buttonbox_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,:3]

    # reset buttonbox to unpressed
    buttonbox_dof_idxs_long = (self.franka_dof_start_idx[env_ids]+self.num_franka_dofs).flatten().to(dtype=torch.long)

    all_pos = torch.zeros_like(self.dof_state)
    all_pos[...,0][buttonbox_dof_idxs_long] = self.buttonbox_dof_upper_limits
    all_pos[...,1][buttonbox_dof_idxs_long] = 0.0

    self.dof_state[...,0].flatten().index_copy_(0,buttonbox_dof_idxs_long.flatten(),all_pos[...,0][buttonbox_dof_idxs_long].flatten())
    self.dof_state[...,1].flatten().index_copy_(0,buttonbox_dof_idxs_long.flatten(),all_pos[...,1][buttonbox_dof_idxs_long].flatten())

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return torch.cat((self.franka_actor_idx[env_ids], buttonbox_multi_env_ids_int32),dim=-1), buttonbox_multi_env_ids_int32

