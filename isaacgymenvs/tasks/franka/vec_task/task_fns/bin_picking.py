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
    obj_asset_file = "assets_v2/unified_objects/envs_objA.xml"
    binA_asset_file = "assets_v2/unified_objects/binA.xml"
    binB_asset_file = "assets_v2/unified_objects/binB.xml"
    # cylinder_asset_file = "assets_v2/unified_objects/cylinder.xml"
    cylinder_asset_file = "assets_v2/unified_objects/block.xml"
    
    # create obj asset
    obj_asset_options = gymapi.AssetOptions()
    obj_asset_options.fix_base_link = False
    obj_asset_options.disable_gravity = False
    obj_asset = self.gym.load_asset(self.sim, asset_root, cylinder_asset_file, obj_asset_options )

    # create binA asset
    binA_asset_options = gymapi.AssetOptions()
    binA_asset_options.fix_base_link = True
    binA_asset_options.disable_gravity = False
    binA_asset = self.gym.load_asset(self.sim, asset_root, binA_asset_file, binA_asset_options)

    # create binB asset
    binB_asset_options = gymapi.AssetOptions()
    binB_asset_options.fix_base_link = True
    binB_asset_options.disable_gravity = False
    binB_asset = self.gym.load_asset(self.sim, asset_root, binB_asset_file, binB_asset_options)

    # define start pose for obj (will be reset later)
    obj_height = .04
    obj_start_pose = gymapi.Transform()
    obj_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+obj_height/2)
    obj_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # define start pose for binA (will NOT be reset later)
    binA_height = 0
    binA_start_pose = gymapi.Transform()
    binA_start_pose.p = gymapi.Vec3(.1, .12, self._table_surface_pos[2]+binA_height/2)
    binA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # define start pose for binB (will NOT be reset later)
    binB_height = 0
    binB_start_pose = gymapi.Transform()
    binB_start_pose.p = gymapi.Vec3(.1, -.12, self._table_surface_pos[2]+binB_height/2)
    binB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    
    # ---------------------- Compute aggregate size ----------------------
    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    
    num_object_bodies = self.gym.get_asset_rigid_body_count(obj_asset) + self.gym.get_asset_rigid_body_count(binA_asset) \
                                + self.gym.get_asset_rigid_body_count(binB_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(obj_asset) + self.gym.get_asset_rigid_shape_count(binA_asset) \
                                + self.gym.get_asset_rigid_shape_count(binB_asset)
    num_object_dofs = self.gym.get_asset_dof_count(obj_asset) + self.gym.get_asset_dof_count(binA_asset) \
                                + self.gym.get_asset_dof_count(binB_asset)
    
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    print("num object bodies: ", num_object_bodies)
    print("num object dofs: ", num_object_dofs)
    
    
    # ---------------------- Define goals ----------------------
    # cube uses obj
    # box width is around 0.2, but metaworld uses (.03,.21) in y, 
    # which places the cube too close to the edge for franka to grip and more importantly, cube starts to flips due to collision
    # changed to (.15,.15) to make the cube centered
    obj_low =  (0.05,  0.15, self._table_surface_pos[2] + obj_height/2) 
    obj_high = (0.15,  0.15, self._table_surface_pos[2] + obj_height/2)

    goal_low =  (0.099, -0.1201, self._table_surface_pos[2]+.05) # cube is already at _table_surface_pos[2] + .03, add a little offset
    goal_high = (0.101, -0.1199, self._table_surface_pos[2]+.05)

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
        
        # Create obj
        obj_actor = self.gym.create_actor(env_ptr, obj_asset, obj_start_pose, "obj", i, -1, 0)

        # create binA
        binA_actor = self.gym.create_actor(env_ptr, binA_asset, binA_start_pose, "binA", i, -1, 0)

        # create binB
        binB_actor = self.gym.create_actor(env_ptr, binB_asset, binB_start_pose, "binB", i, -1, 0)
        
        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                    i, 1, 0)

        # if self.aggregate_mode > 0:
        #     self.gym.end_aggregate(env_ptr)

        envs.append(env_ptr)
        frankas.append(franka_actor)
        objects.append(obj_actor)

    # pass these to the main create envs fns
    num_task_actor = 3  # obj, binA, binB
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    return envs, frankas, objects, random_reset_space, num_task_actor, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env
    
    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    obj_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    obj_quat = self.root_state_tensor[multi_env_ids_int32,3:7]

    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    binA_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    binA_quat = self.root_state_tensor[multi_env_ids_int32,3:7]

    # print("obj_pos: ", obj_pos[0])
    # print('target_pos', self.target_pos[0])
    # print('distance', torch.norm(obj_pos - self.target_pos,dim=-1)[0])
    
    return torch.cat([
        obj_pos,
        obj_quat,
        binA_pos,
        binA_quat
    ], dim=-1)


@torch.jit.script
def compute_reward(
        reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, franka_dof_pos: torch.Tensor,
        franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, max_episode_length: float, 
        init_tcp: torch.Tensor, target_pos: torch.Tensor, obj_pos: torch.Tensor, obj_init_pos: torch.Tensor, specialized_kwargs: Dict[str,torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    TARGET_RADIUS = .05
    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2
    target_to_obj = torch.norm(obj_pos - target_pos,dim=-1)
    target_to_obj_init = torch.norm(obj_init_pos - target_pos,dim=-1)

    in_place = tolerance(
        target_to_obj,
        bounds=(0.0, TARGET_RADIUS),
        margin=target_to_obj_init,
        sigmoid='long_tail',
    )

    threshold = .03
    radii = torch.hstack([
        torch.norm(tcp[:,:2]-obj_init_pos[:,:2],dim=-1).unsqueeze(-1),
        torch.norm(tcp[:,:2]-target_pos[:,:2],dim=-1).unsqueeze(-1),
    ])
    # floor is a *pair* of 3D funnels centered on (1) the object's initial
    # position and (2) the desired final position
    TABLE_Z = 1.0270
    floor = torch.min(torch.where(radii > threshold,.02 * torch.log(radii-threshold) + (TABLE_Z + 0.2),TABLE_Z),dim=-1).values

    # prevent the hand from running into the edge of the bins by keeping
    # it above the "floor"
    above_floor_tolerance = tolerance(
                                torch.maximum(floor-tcp[:,2],torch.zeros_like(floor)),
                                bounds=(0.0, 0.1),
                                margin=torch.ones_like(floor) * .05,
                                sigmoid='long_tail',
                            )
    
    above_floor = torch.where(tcp[:,2] >= floor, 1.0, above_floor_tolerance)

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

    rewards = hamacher_product(object_grasped,in_place)
    near_object = torch.norm(obj_pos-tcp,dim=-1) < .04

    gripper_distance_apart = torch.norm(franka_rfinger_pos-franka_lfinger_pos,dim=-1)
    tcp_opened = torch.clip(gripper_distance_apart/.095,0.0,1.0)
    
    pinched_without_obj = tcp_opened < .33     # .095 * x = .04 => x = .42, .33 is more lenient
    lifted = (obj_pos[:,2] - .02) > obj_init_pos[:,2] 
    
    # increase reward when properly grabbed obj
    grasp_success = near_object & lifted & (tcp_opened > 0) # using pinch_without_obj is too strict and leads to very slow convergence

    # uncomment below to use original metaworld dense reward below
    # rewards = torch.where(grasp_success, rewards + 1.0 + 5.0 * hamacher_product(above_floor,in_place), rewards) * .01

    # the above_floor in the dense reward above creates instability and is not needed. 
    # in_place is enough to guide the hand, and matches the structure of the other pick place tasks
    rewards = torch.where(grasp_success, rewards + 1.0 + 5.0 * in_place, rewards) * .01

    success = target_to_obj < TARGET_RADIUS
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
    self.target_pos[env_ids] = to_torch(self.last_rand_vecs[env_ids,3:],device=self.device)
    
    # get random cube positions
    self.obj_init_pos[env_ids] =  to_torch(self.last_rand_vecs[env_ids,:3],device=self.device)  

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
    obj_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    self.root_state_tensor[obj_multi_env_ids_int32,:3] = self.obj_init_pos[env_ids] + to_torch([0,0,.02],device=self.device)
    self.root_state_tensor[obj_multi_env_ids_int32,3:7] = to_torch([0,0,0,1],device=self.device)
    self.root_state_tensor[obj_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return self.franka_actor_idx[env_ids], obj_multi_env_ids_int32

