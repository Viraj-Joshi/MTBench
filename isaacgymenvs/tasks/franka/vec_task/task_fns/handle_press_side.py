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
    handle_press_asset_file = "assets_v2/unified_objects/handle_press.xml"

    # load handle_press asset
    handle_press_asset_options = gymapi.AssetOptions()
    handle_press_asset_options.fix_base_link = True     
    handle_press_asset_options.disable_gravity = True
    handle_press_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    handle_press_asset_options.replace_cylinder_with_capsule = True
    handle_press_asset = self.gym.load_asset(self.sim, asset_root, handle_press_asset_file, handle_press_asset_options)

    # set handle_press dof properties
    handle_press_dof_props = self.gym.get_asset_dof_properties(handle_press_asset)
    num_handle_press_dofs = self.gym.get_asset_dof_count(handle_press_asset)
    
    handle_press_dof_lower_limits = []
    handle_press_dof_upper_limits = []
    for i in range(num_handle_press_dofs):
        handle_press_dof_lower_limits.append(handle_press_dof_props['lower'][i])
        handle_press_dof_upper_limits.append(handle_press_dof_props['upper'][i])
        # door_dof_props['damping'][i] = 10.0
    
    self.handle_press_dof_lower_limits = to_torch(handle_press_dof_lower_limits, device=self.device)
    self.handle_press_dof_upper_limits = to_torch(handle_press_dof_upper_limits, device=self.device)

    # Define start pose for handle_press (going to be reset anyway)
    handle_press_height = 0
    handle_press_start_pose = gymapi.Transform()
    handle_press_start_pose.p = gymapi.Vec3(.22,-.3,self._table_surface_pos[2]+handle_press_height/2)
    handle_press_start_pose.r = gymapi.Quat( 0, 0, 0, 1)


    # ---------------------- Compute aggregate size ----------------------
    num_object_bodies = self.gym.get_asset_rigid_body_count(handle_press_asset)
    num_object_dofs = self.gym.get_asset_dof_count(handle_press_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(handle_press_asset)

    print("num bodies: ", num_object_bodies)
    print("num obj dofs: ", num_object_dofs)

    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    # check whether they are the same across the tasks
    table_pos = [0.0, 0.0, 1.0]
    table_thickness = 0.054

    # ---------------------- Define goals ----------------------    
    # obj is used for handle_press placement
    goal_low = (0, 0, 0)
    goal_high = (0, 0, 0)
    obj_low =  (.05, .25, self._table_surface_pos[2]+handle_press_height/2)
    obj_high = (.15, .35, self._table_surface_pos[2]+handle_press_height/2)

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

        handle_press_pose = handle_press_start_pose
        handle_press_actor = self.gym.create_actor(env_ptr, handle_press_asset, handle_press_pose, "handle_press", i, -1, 0)
        self.gym.set_actor_dof_properties(env_ptr, handle_press_actor, handle_press_dof_props)

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
        objects.append(handle_press_actor)
        
    # pass these to the main create envs fns
    num_task_actors = 1  # this sence only has 1 actor except for franka and table
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies

    return envs, frankas, objects, random_reset_space, num_task_actors, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env

    # handleStart rigid body idx is offset from last franka rigid body
    handle_start_rigid_body_idx = \
        self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 4
    handle_start_rigid_body_states = self.rigid_body_states[handle_start_rigid_body_idx].view(-1, 13)
    obj_pos = handle_start_rigid_body_states[:, 0:3]
    obj_quat = handle_start_rigid_body_states[:, 3:7]
      
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

    TARGET_RADIUS = .02

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2
    
    target_to_obj = torch.norm(obj_pos - target_pos,dim=-1)
    target_to_obj_init = torch.norm(target_pos- obj_init_pos,dim=-1)

    in_place = tolerance(
        target_to_obj,
        bounds=(0.0, TARGET_RADIUS),
        margin=torch.abs(target_to_obj_init-TARGET_RADIUS),
        sigmoid="long_tail",
    )

    handle_radius = .02
    tcp_to_obj = torch.norm(obj_pos - tcp,dim=-1)
    tcp_to_obj_init = torch.norm(obj_pos - tcp_init,dim=-1)
    reach = tolerance(
        tcp_to_obj,
        bounds=(0.0, handle_radius),
        margin=torch.abs(tcp_to_obj_init - handle_radius),
        sigmoid="long_tail",
    )

    rewards = hamacher_product(reach, in_place) * .01
    success = (target_to_obj <= TARGET_RADIUS) # & (tcp_to_obj <= .02)
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

    # get obj init pos (should be the same handleStart body position)
    self.obj_init_pos[env_ids] = self.last_rand_vecs[env_ids,:3].clone()
    self.obj_init_pos[env_ids,1] += -.216
    self.obj_init_pos[env_ids,2] += .172 # an extra .0025 would match goalPress from xml but that would be due to the radius of the sphere (.005/2)
    
    # the target is the goalPress body position
    self.target_pos[env_ids] = self.obj_init_pos[env_ids].clone()
    self.target_pos[env_ids,2] += -(.172-.075)

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
    self.root_state_tensor[obj_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,:3]

    # reset obj to unpressed
    obj_dof_idxs_long = (self.franka_dof_start_idx[env_ids]+self.num_franka_dofs).flatten().to(dtype=torch.long)

    all_pos = torch.zeros_like(self.dof_state)
    all_pos[...,0][obj_dof_idxs_long] = self.handle_press_dof_upper_limits
    all_pos[...,1][obj_dof_idxs_long] = 0.0

    self.dof_state[...,0].flatten().index_copy_(0,obj_dof_idxs_long.flatten(),all_pos[...,0][obj_dof_idxs_long].flatten())
    self.dof_state[...,1].flatten().index_copy_(0,obj_dof_idxs_long.flatten(),all_pos[...,1][obj_dof_idxs_long].flatten())

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return torch.cat((self.franka_actor_idx[env_ids],self.franka_actor_idx[env_ids]+1),dim=-1), obj_multi_env_ids_int32

