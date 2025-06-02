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
    basketball_asset_file = "assets_v2/unified_objects/basketball.xml"
    # basketball_asset_file = "assets_v2/unified_objects/block.xml"
    basket_hoop_asset_file = "assets_v2/unified_objects/basket_hoop.xml"

    # create envs asset
    basketball_asset_options = gymapi.AssetOptions()
    basketball_asset_options.fix_base_link = False
    basketball_asset_options.disable_gravity = False
    basketball_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    basketball_asset = self.gym.load_asset(self.sim, asset_root, basketball_asset_file, basketball_asset_options )
    # basketball_asset = self.gym.create_sphere(self.sim, 0.03, basketball_asset_options)

    # create envs hoop asset
    basket_hoop_asset_options = gymapi.AssetOptions()
    basket_hoop_asset_options.fix_base_link = True
    basket_hoop_asset = self.gym.load_asset(self.sim, asset_root, basket_hoop_asset_file, basket_hoop_asset_options)

    # define start pose for envs (will be reset later)
    envs_height = .04
    envs_start_pose = gymapi.Transform()
    envs_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+envs_height/2)
    envs_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # define start pose for basket hoop (will NOT be reset later)
    basket_hoop_height = 0
    basket_hoop_start_pose = gymapi.Transform()
    basket_hoop_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+basket_hoop_height/2)
    basket_hoop_start_pose.r = gymapi.Quat(0, 0, -0.7068252, 0.7073883)
    
    
    # ---------------------- Compute aggregate size ----------------------
    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    
    num_object_bodies = self.gym.get_asset_rigid_body_count(basketball_asset)  + self.gym.get_asset_rigid_body_count(basket_hoop_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(basketball_asset) + self.gym.get_asset_rigid_shape_count(basket_hoop_asset)
    num_object_dofs = self.gym.get_asset_dof_count(basketball_asset) + self.gym.get_asset_dof_count(basket_hoop_asset)
    
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    print("num object bodies: ", num_object_bodies)
    print("num object dofs: ", num_object_dofs)
    
    
    # ---------------------- Define goals ----------------------
    # envs uses obj and basket_hoop uses goal
    obj_low =  (0.0, -.1, self._table_surface_pos[2]+envs_height/2)
    obj_high = (0.1, .1, self._table_surface_pos[2]+envs_height/2)

    goal_low =  (.25, -.1, self._table_surface_pos[2])
    goal_high = (.30,  .1, self._table_surface_pos[2])

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
        
        # Create basketball
        basketball_actor = self.gym.create_actor(env_ptr, basketball_asset, envs_start_pose, "bsktball", i, -1, 0)
        # basketball_actor_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, basketball_actor)
        # for basketball_actor_shape_prop in basketball_actor_shape_props:
        #     basketball_actor_shape_prop.friction = 1.0
        #     basketball_actor_shape_prop.rolling_friction = 1.0
        #     basketball_actor_shape_prop.torsion_friction = 1.0
        
        # self.gym.set_actor_rigid_shape_properties(env_ptr, basketball_actor, basketball_actor_shape_props)

        # create hoop
        basket_hoop_actor = self.gym.create_actor(env_ptr, basket_hoop_asset, basket_hoop_start_pose, "basket_hoop", i, -1, 0)
        
        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, -1, 0)
        # table_actor_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)
        # for table_actor_shape_prop in table_actor_shape_props:
        #     table_actor_shape_prop.friction = 1.0
        #     table_actor_shape_prop.rolling_friction = 1.0
        #     table_actor_shape_prop.torsion_friction = 1.0             

        # self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_actor_shape_props)

        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                    i, 1, 0)

        # if self.aggregate_mode > 0:
        #     self.gym.end_aggregate(env_ptr)

        envs.append(env_ptr)
        frankas.append(franka_actor)
        objects.append(basketball_actor)

    # pass these to the main create envs fns
    num_task_actor = 2  # basketball and hoop
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    return envs, frankas, objects, random_reset_space, num_task_actor, num_task_dofs, num_task_bodies



def compute_observations(env, env_ids):
    self = env
    
    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    obj_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    obj_quat = self.root_state_tensor[multi_env_ids_int32,3:7]

    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    basket_hoop_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    basket_hoop_quat = self.root_state_tensor[multi_env_ids_int32,3:7]

    return torch.cat([
        obj_pos,
        obj_quat,
        basket_hoop_pos,
        basket_hoop_quat
    ], dim=-1)


@torch.jit.script
def compute_reward(
        reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, franka_dof_pos: torch.Tensor,
        franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, max_episode_length: float, 
        init_tcp: torch.Tensor, target_pos: torch.Tensor, obj_pos: torch.Tensor, obj_init_pos: torch.Tensor, specialized_kwargs: Dict[str,torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    TARGET_RADIUS = .08
    scale = specialized_kwargs["scale"]

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2

    target_to_obj = torch.norm((obj_pos-target_pos)*scale,dim=-1)
    target_to_obj_init = torch.norm((obj_init_pos - target_pos)*scale,dim=-1)

    in_place = tolerance(
        target_to_obj,
        bounds=(0.0, TARGET_RADIUS),
        margin=target_to_obj_init,
        sigmoid="long_tail",
    )

    # create a normalized 0 to 1 measurement of how open the gripper is, where 1 is fully open and 0 is fully closed
    gripper_distance_apart = torch.norm(franka_rfinger_pos-franka_lfinger_pos,dim=-1)
    tcp_opened = torch.clip(gripper_distance_apart/.095,0.0,1.0)
    
    tcp_to_obj = torch.norm(obj_pos - tcp,dim=-1)

    object_grasped = _gripper_caging_reward(
        obj_pos,
        franka_lfinger_pos,
        franka_rfinger_pos,
        tcp,
        init_tcp,
        actions,
        obj_init_pos,
        obj_radius=.025,
        object_reach_radius=.01,
        pad_success_thresh=.06,
        xz_thresh=.005,
        medium_density=True
    )

    in_place_and_object_grasped = hamacher_product(
        object_grasped, in_place
    )

    rewards = in_place_and_object_grasped
    rewards = torch.where((tcp_to_obj < .02) & (tcp_opened > 0 ) & ((obj_pos[:,2]-.01) > obj_init_pos[:,2]), rewards + 1.0 + 5.0 * in_place, rewards)*.01

    success = target_to_obj < TARGET_RADIUS
    rewards = torch.where(success, 10, rewards)

    # reset if max length reached or success
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

    # some offsets to get to hoop
    self.target_pos[env_ids] = self.last_rand_vecs[env_ids,3:].clone() + to_torch([-.083,0,.25],device=self.device)

    self.obj_init_pos[env_ids] =  self.last_rand_vecs[env_ids,:3]

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
    obj_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    self.root_state_tensor[obj_env_ids_int32,:3] = self.obj_init_pos[env_ids]
    self.root_state_tensor[obj_env_ids_int32,3:7] = to_torch([0,0,0,1],device=self.device)
    self.root_state_tensor[obj_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    basket_hoop_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    self.root_state_tensor[basket_hoop_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,3:]

    actor_multi_env_ids_int32 = torch.cat((obj_env_ids_int32, basket_hoop_multi_env_ids_int32),dim=-1).flatten().to(dtype=torch.int32)

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return self.franka_actor_idx[env_ids], actor_multi_env_ids_int32

