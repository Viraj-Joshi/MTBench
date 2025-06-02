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
    soccer_ball_asset_file = "assets_v2/unified_objects/soccer_ball.xml"
    soccer_goal_asset_file = "assets_v2/unified_objects/soccer_goal.xml"

    # load soccer_ball asset
    soccer_ball_asset_options = gymapi.AssetOptions()
    soccer_ball_asset_options.fix_base_link = False
    soccer_ball_asset_options.disable_gravity = False
    soccer_ball_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    soccer_ball_asset = self.gym.load_asset(self.sim, asset_root, soccer_ball_asset_file, soccer_ball_asset_options)

    # load soccer_goal asset
    soccer_goal_asset_options = gymapi.AssetOptions()
    soccer_goal_asset_options.fix_base_link = True
    soccer_goal_asset_options.disable_gravity = False
    soccer_goal_asset = self.gym.load_asset(self.sim, asset_root, soccer_goal_asset_file, soccer_goal_asset_options )

    # Define start pose for soccer_ball (going to be reset anyway)
    soccer_ball_height = 0.06
    soccer_ball_start_pose = gymapi.Transform()
    soccer_ball_start_pose.p = gymapi.Vec3(.22,-.3,self._table_surface_pos[2]+soccer_ball_height/2)
    soccer_ball_start_pose.r = gymapi.Quat( 0, 0, 0, 1)

    # define start pose for soccer_goal (will be reset later)
    soccer_goal_height = 0
    soccer_goal_start_pose = gymapi.Transform()
    soccer_goal_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+soccer_goal_height/2)
    soccer_goal_start_pose.r = gymapi.Quat(0, 0, -0.7068252, 0.7073883)
    
    
    # ---------------------- Compute aggregate size ----------------------
    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    
    num_object_bodies = self.gym.get_asset_rigid_body_count(soccer_ball_asset)  + self.gym.get_asset_rigid_body_count(soccer_goal_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(soccer_ball_asset) + self.gym.get_asset_rigid_shape_count(soccer_goal_asset)
    num_object_dofs = self.gym.get_asset_dof_count(soccer_ball_asset) + self.gym.get_asset_dof_count(soccer_goal_asset)
    
    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    print("num object bodies: ", num_object_bodies)
    print("num object dofs: ", num_object_dofs)
    
    
    # ---------------------- Define goals ----------------------
    # obj is used for soccer_goal placement and goal is the target pos
    obj_low =  (0,  -.1, self._table_surface_pos[2] + soccer_ball_height/2)
    obj_high = (0.1, .1, self._table_surface_pos[2] + soccer_ball_height/2)
    goal_low =  (.2, -.1, self._table_surface_pos[2])
    goal_high = (.3,  .1, self._table_surface_pos[2])

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
        
        soccer_ball_pose = soccer_ball_start_pose
        soccer_ball_actor = self.gym.create_actor(env_ptr, soccer_ball_asset, soccer_ball_pose, "soccer_ball", i, -1, 0)

        soccer_goal_actor = self.gym.create_actor(env_ptr, soccer_goal_asset, soccer_goal_start_pose, "soccer_goal", i, -1, 0)
        
        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                    i, 1, 0)

        # if self.aggregate_mode > 0:
        #     self.gym.end_aggregate(env_ptr)

        envs.append(env_ptr)
        frankas.append(franka_actor)
        objects.append(soccer_ball_actor)

    # pass these to the main create envs fns
    num_task_actor = 2  # ball and net
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    return envs, frankas, objects, random_reset_space, num_task_actor, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env
    
    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    soccer_ball_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    soccer_ball_rot = self.root_state_tensor[multi_env_ids_int32,3:7]

    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    soccer_goal_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    soccer_goal_rot = self.root_state_tensor[multi_env_ids_int32,3:7]

    return torch.cat([
        soccer_ball_pos,
        soccer_ball_rot,
        soccer_goal_pos,
        soccer_goal_rot
    ], dim=-1)

@torch.jit.script
def compute_reward(
        reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, franka_dof_pos: torch.Tensor,
        franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, max_episode_length: float, 
        init_tcp: torch.Tensor, target_pos: torch.Tensor, obj_pos: torch.Tensor, obj_init_pos: torch.Tensor, specialized_kwargs: Dict[str,torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    TARGET_RADIUS = .07
    scale = specialized_kwargs['scale']

    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2
    tcp_to_obj = torch.norm(tcp-obj_pos,dim=-1)

    target_to_obj = torch.norm((obj_pos - target_pos)*scale,dim=-1)
    target_to_obj_init = torch.norm((obj_pos - obj_init_pos)*scale,dim=-1)

    in_place = tolerance(
        target_to_obj,
        bounds= (0.0,TARGET_RADIUS),
        margin=target_to_obj_init,
        sigmoid='long_tail'
    )

    goal_line = target_pos[:,0].unsqueeze(-1)
    in_place = torch.where((obj_pos[:,0] >= goal_line[:,0]) & (torch.abs(obj_pos[:,1] - target_pos[:,1]) > 0.1),
                            torch.clamp(in_place - 2 * ((obj_pos[:,0] - goal_line[:,0])/ (1-goal_line[:,0])), 0.0, 1.0),
                            in_place)
    
    # changed float parameters from metaworld
    object_grasped = _gripper_caging_reward(
        obj_pos,
        franka_lfinger_pos,
        franka_rfinger_pos,
        tcp,
        init_tcp,
        actions,
        obj_init_pos,
        obj_radius=.013,
        object_reach_radius=.01,
        pad_success_thresh=.05,
        xz_thresh=.005,
        medium_density=True
    )

    # object_grasped = _gripper_caging_reward(
    #     obj_pos,
    #     franka_lfinger_pos,
    #     franka_rfinger_pos,
    #     tcp,
    #     init_left_pad,
    #     init_right_pad,
    #     actions,
    #     obj_init_pos,
    #     obj_radius=.013,
    # )
    
    rewards = ((3 * object_grasped) + (6.5 * in_place)) * .01
    
    success = (target_to_obj < TARGET_RADIUS)
    rewards = torch.where(success, 10, rewards)

    # Compute resets
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

    # get obj init pos (is the soccer_ball)
    self.obj_init_pos[env_ids] = self.last_rand_vecs[env_ids,:3]
    
    # target is offset from the soccer goal
    self.target_pos[env_ids] = self.last_rand_vecs[env_ids,3:].clone() + to_torch([-.1, 0, .013],device=self.device)      

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
    self.root_state_tensor[obj_multi_env_ids_int32,:3] = self.obj_init_pos[env_ids]
    self.root_state_tensor[obj_multi_env_ids_int32,3:7] = to_torch([0,0,0,1],device=self.device)
    self.root_state_tensor[obj_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    soccer_goal_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    self.root_state_tensor[soccer_goal_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,3:]

    actor_multi_env_ids_int32 = torch.cat((obj_multi_env_ids_int32,soccer_goal_multi_env_ids_int32),dim=-1).flatten().to(dtype=torch.int32)

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    return self.franka_actor_idx[env_ids],actor_multi_env_ids_int32

