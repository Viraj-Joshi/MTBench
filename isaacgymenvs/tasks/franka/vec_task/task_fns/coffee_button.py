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
    coffee_machine_asset_file = "assets_v2/unified_objects/coffee.xml"
    mug_asset_file = "assets_v2/unified_objects/mug.xml"

    # load coffee_machine asset
    coffee_asset_options = gymapi.AssetOptions()
    coffee_asset_options.fix_base_link = True
    coffee_asset_options.disable_gravity = False
    coffee_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    coffee_machine_asset = self.gym.load_asset(self.sim, asset_root, coffee_machine_asset_file, coffee_asset_options)

    # load mug asset
    mug_asset_options = gymapi.AssetOptions()
    mug_asset_options.fix_base_link = False
    mug_asset_options.disable_gravity = False
    mug_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    mug_asset = self.gym.load_asset(self.sim, asset_root, mug_asset_file, mug_asset_options)

    # set door dof properties
    coffee_machine_dof_props = self.gym.get_asset_dof_properties(coffee_machine_asset)
    num_coffee_machine_dofs = self.gym.get_asset_dof_count(coffee_machine_asset)
    
    coffee_machine_dof_lower_limits = []
    coffee_machine_dof_upper_limits = []
    for i in range(num_coffee_machine_dofs):
        coffee_machine_dof_lower_limits.append(coffee_machine_dof_props['lower'][i])
        coffee_machine_dof_upper_limits.append(coffee_machine_dof_props['upper'][i])
        # door_dof_props['damping'][i] = 10.0
    
    self.coffee_machine_dof_lower_limits = to_torch(coffee_machine_dof_lower_limits, device=self.device)
    self.coffee_machine_dof_upper_limits = to_torch(coffee_machine_dof_upper_limits, device=self.device)

    # Define start pose for coffee_machine (going to be reset anyway)
    coffee_machine_height = 0
    coffee_machine_button_start_pose = gymapi.Transform()
    coffee_machine_button_start_pose.p = gymapi.Vec3(.22,-.3,self._table_surface_pos[2]+coffee_machine_height/2)
    coffee_machine_button_start_pose.r = gymapi.Quat( 0, 0, -0.7068252, 0.7073883)
    
    # define start pose for mug (will be reset)
    mug_height = 0
    mug_start_pose = gymapi.Transform()
    mug_start_pose.p = gymapi.Vec3(.1,-.3,self._table_surface_pos[2]+mug_height/2)
    mug_start_pose.r = gymapi.Quat( 0, 0, 0, 1)


    # ---------------------- Compute aggregate size ----------------------
    num_object_bodies = self.gym.get_asset_rigid_body_count(coffee_machine_asset) + self.gym.get_asset_rigid_body_count(mug_asset)
    num_object_dofs = self.gym.get_asset_dof_count(coffee_machine_asset) + self.gym.get_asset_dof_count(mug_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(coffee_machine_asset) + self.gym.get_asset_rigid_shape_count(mug_asset)

    print("num bodies: ", num_object_bodies)
    print("num coffee machine dofs: ", num_object_dofs)

    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    # check whether they are the same across the tasks
    table_pos = [0.0, 0.0, 1.0]
    table_thickness = 0.054

    # ---------------------- Define goals ----------------------    
    # obj is used for coffee_machine placement
    obj_low =  (.2, -.1, self._table_surface_pos[2]+coffee_machine_height/2) 
    obj_high = (.25, .1, self._table_surface_pos[2]+coffee_machine_height/2) # changed from .3 to .25
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

        coffee_machine_pose = coffee_machine_button_start_pose
        coffee_machine_actor = self.gym.create_actor(env_ptr, coffee_machine_asset, coffee_machine_pose, "coffee_machine", i, -1, 0)
        self.gym.set_actor_dof_properties(env_ptr, coffee_machine_actor, coffee_machine_dof_props)

        mug_actor = self.gym.create_actor(env_ptr, mug_asset, mug_start_pose, "mug", i, -1, 0)

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
        objects.append(coffee_machine_actor)
        
    # pass these to the main create envs fns
    num_task_actors = 2  # mug and coffee_machine
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    
    return envs, frankas, objects, random_reset_space, num_task_actors, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env

    coffee_machine_button_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 4
    coffee_machine_button_rigid_body_states = self.rigid_body_states[coffee_machine_button_rigid_body_idx].view(-1, 13)
    coffee_machine_button_pos = coffee_machine_button_rigid_body_states[:, 0:3]
    coffee_machine_button_rot = coffee_machine_button_rigid_body_states[:, 3:7]

    mug_pos = self.root_state_tensor[self.franka_actor_idx[env_ids]+2,:3]
    mug_rot = self.root_state_tensor[self.franka_actor_idx[env_ids]+2,3:7]
  
    return torch.cat([
        coffee_machine_button_pos,
        coffee_machine_button_rot,
        mug_pos,
        mug_rot,
    ], dim=-1)


@torch.jit.script
def compute_reward(
    reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, franka_dof_pos: torch.Tensor,
    franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, max_episode_length: float, 
    tcp_init: torch.Tensor, target_pos: torch.Tensor, obj_pos: torch.Tensor, obj_init_pos: torch.Tensor, specialized_kwargs: Dict[str,torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2
    
    tcp_to_obj = torch.norm(obj_pos - tcp,dim=-1)
    tcp_to_obj_init = torch.norm(obj_pos - tcp_init,dim=-1)
    obj_to_target = torch.abs(obj_pos[:,0] - target_pos[:,0])

    gripper_distance_apart = torch.norm(franka_rfinger_pos-franka_lfinger_pos,dim=-1)
    tcp_opened = torch.clip(gripper_distance_apart/.095,0.0,1.0)

    tcp_closed = 1-tcp_opened
    near_button = tolerance(
        tcp_to_obj,
        bounds=(0.0, 0.05),
        margin=tcp_to_obj_init,
        sigmoid="long_tail",
    )

    button_pressed = tolerance(
        obj_to_target,
        bounds=(0.0, 0.005),
        margin=torch.ones_like(tcp_closed)*.03,
        sigmoid="long_tail",
    )

    rewards = 2 * hamacher_product(tcp_closed, near_button)
    rewards = torch.where((tcp_to_obj <= 0.05) & (tcp_closed>=.95), rewards + 8*button_pressed, rewards) * .01
    
    success = (obj_to_target <= 0.01) & (tcp_to_obj < .02)
    rewards = torch.where(success, 10.0, rewards)
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

    # get obj init pos
    self.obj_init_pos[env_ids] = self.last_rand_vecs[env_ids,:3].clone()
    self.obj_init_pos[env_ids,0] += -.22
    self.obj_init_pos[env_ids,2] += .3
    
    # the target is just the obj init pos plus the coffee_machine button pressed along the x-axis (+.03)
    self.target_pos[env_ids] = self.obj_init_pos[env_ids].clone()
    self.target_pos[env_ids,0] += .03 

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
    coffee_machine_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    self.root_state_tensor[coffee_machine_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,:3]

    mug_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    self.root_state_tensor[mug_multi_env_ids_int32,:3] = self.obj_init_pos[env_ids] + to_torch([-.15,0,0],device=self.device)
    self.root_state_tensor[mug_multi_env_ids_int32,2] = self._table_surface_pos[2] + .005  
    self.root_state_tensor[mug_multi_env_ids_int32,3:7] = to_torch([0, 0, 0.7071068, 0.7071068], device=self.device)
    self.root_state_tensor[mug_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    actor_multi_env_ids_int32 = torch.cat((coffee_machine_multi_env_ids_int32,mug_multi_env_ids_int32),dim=-1)
    
    # reset coffee button to unpressed
    coffee_machine_dof_idxs_long = (self.franka_dof_start_idx[env_ids]+self.num_franka_dofs).flatten().to(dtype=torch.long)

    all_pos = torch.zeros_like(self.dof_state)
    all_pos[...,0][coffee_machine_dof_idxs_long] = self.coffee_machine_dof_lower_limits
    all_pos[...,1][coffee_machine_dof_idxs_long] = 0.0

    self.dof_state[...,0].flatten().index_copy_(0,coffee_machine_dof_idxs_long.flatten(),all_pos[...,0][coffee_machine_dof_idxs_long].flatten())
    self.dof_state[...,1].flatten().index_copy_(0,coffee_machine_dof_idxs_long.flatten(),all_pos[...,1][coffee_machine_dof_idxs_long].flatten())

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return torch.cat((self.franka_actor_idx[env_ids],self.franka_actor_idx[env_ids]+1),dim=-1), actor_multi_env_ids_int32

