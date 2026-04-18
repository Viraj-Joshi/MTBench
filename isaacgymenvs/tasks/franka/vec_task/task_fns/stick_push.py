import os
from typing import Dict, Tuple

import torch
import numpy as np
from gym import spaces

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch
from isaacgymenvs.tasks.reward_utils import _gripper_caging_reward, hamacher_product,_sigmoids

_DEFAULT_VALUE_AT_MARGIN = .1
@torch.jit.script
def tolerance_fixed(
    x,
    margin,
    bounds=(0.0, 0.0),
    sigmoid="gaussian",
    value_at_margin=_DEFAULT_VALUE_AT_MARGIN,
):
    # type: (Tensor, Tensor, Tuple[float,float], str, float) -> Tensor
    """Fixed tolerance that correctly returns 1.0 when x is within bounds and margin > 0."""
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if (margin < 0).any():
        b = margin < 0
        raise ValueError(f"`margin` must be non-negative. Neg at indices {b.nonzero()} with values {margin[b.nonzero()]}")

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    d = torch.where(x < lower, lower - x, x - upper) / torch.clamp(margin, min=1e-8)
    sigmoid_value = _sigmoids(d, value_at_margin, sigmoid)
    value = torch.where(in_bounds, 1.0, torch.where(margin > 0, sigmoid_value, 0.0))
    return value

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
    thermos_asset_file = "assets_v2/unified_objects/thermos.xml"
    stick_asset_file = "assets_v2/unified_objects/stick_l.xml"

    # load thermos asset
    thermos_asset_options = gymapi.AssetOptions()
    thermos_asset_options.fix_base_link = True
    thermos_asset_options.collapse_fixed_joints = False
    thermos_asset_options.disable_gravity = False
    thermos_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    thermos_asset_options.replace_cylinder_with_capsule = True
    thermos_asset = self.gym.load_asset(self.sim, asset_root, thermos_asset_file, thermos_asset_options)

    # load stick asset
    stick_asset_options = gymapi.AssetOptions()
    stick_asset_options.fix_base_link = False
    stick_asset_options.disable_gravity = False
    stick_asset = self.gym.load_asset(self.sim, asset_root, stick_asset_file, stick_asset_options )

    self.num_thermos_dofs = self.gym.get_asset_dof_count(thermos_asset)

    # set thermos dof properties
    thermos_dof_props = self.gym.get_asset_dof_properties(thermos_asset)
    self.thermos_dof_lower_limits = []
    self.thermos_dof_upper_limits = []
    for i in range(self.num_thermos_dofs):
        self.thermos_dof_lower_limits.append(thermos_dof_props['lower'][i])
        self.thermos_dof_upper_limits.append(thermos_dof_props['upper'][i])
        # thermos_dof_props['damping'][i] = 10.0
    
    self.thermos_dof_lower_limits = to_torch(self.thermos_dof_lower_limits, device=self.device)
    self.thermos_dof_upper_limits = to_torch(self.thermos_dof_upper_limits, device=self.device)

    # Define start pose for thermos (going to be reset anyway)
    thermos_height = 0
    thermos_start_pose = gymapi.Transform()
    thermos_start_pose.p = gymapi.Vec3(.22,-.3,self._table_surface_pos[2]+thermos_height/2)
    thermos_start_pose.r = gymapi.Quat( 0, 0, 0, 1)

    # define start pose for stick (will be reset later)
    stick_height = .04
    stick_start_pose = gymapi.Transform()
    stick_start_pose.p = gymapi.Vec3(.3, 0, self._table_surface_pos[2]+stick_height/2)
    stick_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    
    
    # ---------------------- Compute aggregate size ----------------------
    num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
    
    num_object_bodies = self.gym.get_asset_rigid_body_count(stick_asset)  + self.gym.get_asset_rigid_body_count(thermos_asset)
    num_object_shapes = self.gym.get_asset_rigid_shape_count(stick_asset) + self.gym.get_asset_rigid_shape_count(thermos_asset)
    num_object_dofs = self.gym.get_asset_dof_count(stick_asset) + self.gym.get_asset_dof_count(thermos_asset)
    
    self.stick_push_num_thermos_rigid_bodies = self.gym.get_asset_rigid_body_count(thermos_asset)

    max_agg_bodies = num_franka_bodies + num_object_bodies + 2
    max_agg_shapes = num_franka_shapes + num_object_shapes + 2

    print("num object bodies: ", num_object_bodies)
    print("num object dofs: ", num_object_dofs)
    
    
    # ---------------------- Define goals ----------------------
    # obj is used for stick placement and goal is the target pos
    # obj_low =  (-.02, .03, self._table_surface_pos[2]+stick_height/2)
    # obj_high = ( .02, .08, self._table_surface_pos[2]+stick_height/2)
    # goal_low =  (-0.05, -.4, self._table_surface_pos[2]+.110)
    # goal_high = ( 0.00, -.4, self._table_surface_pos[2]+.110)

    obj_low =  (.08, .03, self._table_surface_pos[2]+stick_height/2)
    obj_high = (.12, .08, self._table_surface_pos[2]+stick_height/2)
    goal_low =  (0.05, -.4, self._table_surface_pos[2]+.170)
    goal_high = (0.10, -.4, self._table_surface_pos[2]+.170)

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
        
        thermos_pose = thermos_start_pose
        thermos_actor = self.gym.create_actor(env_ptr, thermos_asset, thermos_pose, "thermos", i, -1, 0)
        self.gym.set_actor_dof_properties(env_ptr, thermos_actor, thermos_dof_props)

        stick_actor = self.gym.create_actor(env_ptr, stick_asset, stick_start_pose, "stick", i, -1, 0)
        
        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                    i, 1, 0)

        # if self.aggregate_mode > 0:
        #     self.gym.end_aggregate(env_ptr)

        envs.append(env_ptr)
        frankas.append(franka_actor)
        objects.append(stick_asset)

    # pass these to the main create envs fns
    num_task_actor = 2  # thermos and stick
    num_task_dofs = num_object_dofs
    num_task_bodies = num_object_bodies
    return envs, frankas, objects, random_reset_space, num_task_actor, num_task_dofs, num_task_bodies


def compute_observations(env, env_ids):
    self = env
    sk = self.specialized_kwargs['stick_push']

    #### Thermos ####
    # Read thermos main body rigid body position directly (offset +2: the body with slide joints and all geoms)
    # (offset +3 is the insertion/handle body used by stick_pull; for pushing we target the main container)
    thermos_body_rigid_body_idx = self.franka_rigid_body_start_idx[env_ids] + self.num_franka_rigid_bodies + 2
    thermos_body_rigid_body_states = self.rigid_body_states[thermos_body_rigid_body_idx].view(-1, 13)
    thermos_body_pos = thermos_body_rigid_body_states[:, :3].clone()
    thermos_body_pos[:, 2] += 0.17  # offset from body frame origin (table level) to upper-container push point
    thermos_body_rot = thermos_body_rigid_body_states[:, 3:7]

    #### Stick ####
    multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    stick_pos = self.root_state_tensor[multi_env_ids_int32,:3]
    stick_rot = self.root_state_tensor[multi_env_ids_int32,3:7]

    # Init positions are set analytically in reset_env() to avoid stale rigid body states at progress_buf==0

    sk['thermos_body_pos'] = thermos_body_pos
    sk['env_ids'] = env_ids

    return torch.cat([
        stick_pos,
        stick_rot,
        thermos_body_pos,
        thermos_body_rot,
    ], dim=-1)


@torch.jit.script
def _stick_is_pushing(stick: torch.Tensor, container_body: torch.Tensor) -> torch.Tensor:
    # Stick is in pushing contact with the container body: aligned in x and z within
    # tight tolerances. The container body z is offset to the upper-container push
    # point so that requiring z alignment forces the wrist to lift clear of the table.
    return torch.logical_and(
        torch.abs(stick[:, 0] - container_body[:, 0]) <= 0.06,
        torch.abs(stick[:, 2] - container_body[:, 2]) <= 0.04,
    )

@torch.jit.script
def compute_reward(
        reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, franka_dof_pos: torch.Tensor,
        franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, max_episode_length: float,
        init_tcp: torch.Tensor, target_pos: torch.Tensor, stick_pos: torch.Tensor, obj_init_pos: torch.Tensor, specialized_kwargs: Dict[str,torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # initialize variables not in standard parameters passed in
    TARGET_RADIUS = .05
    _STICK_TARGET_RADIUS = .05
    tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2
    env_ids = specialized_kwargs['env_ids']
    stick_init_pos = specialized_kwargs['stick_init_pos'][env_ids]

    # define the "handle" as the container body push point (upper container)
    handle = specialized_kwargs['thermos_body_pos']
    handle_to_target = torch.norm(target_pos - handle, dim=-1)

    # define relevant distances from stick
    tcp_to_stick = torch.norm(stick_pos - tcp, dim=-1)
    yz_scaling = specialized_kwargs['yz_scaling']

    # Guide stick center toward the container push point
    stick_to_handle = torch.norm((stick_pos - handle) * yz_scaling, dim=-1)
    stick_to_handle_margin = torch.norm((stick_init_pos - obj_init_pos) * yz_scaling, dim=-1)
    stick_in_place = tolerance_fixed(
        stick_to_handle,
        bounds=(0.0, _STICK_TARGET_RADIUS),
        margin=stick_to_handle_margin,
        sigmoid="long_tail",
    )

    # create a normalized 0 to 1 measurement of how open the gripper is, where 1 is fully open and 0 is fully closed
    gripper_distance_apart = torch.norm(franka_rfinger_pos - franka_lfinger_pos, dim=-1)
    tcp_opened = torch.clip(gripper_distance_apart / .095, 0.0, 1.0)

    object_grasped = _gripper_caging_reward(
        stick_pos,
        franka_lfinger_pos,
        franka_rfinger_pos,
        tcp,
        init_tcp,
        actions,
        stick_init_pos,
        object_reach_radius=0.01,
        obj_radius=0.02,
        pad_success_thresh=0.05,
        xz_thresh=0.005,
        medium_density=True
    )

    grasp_success = (tcp_to_stick < .05) & (tcp_opened > 0) & ((stick_pos[:,2]-.01) > stick_init_pos[:,2])

    object_grasped = torch.where(grasp_success, 1, object_grasped)

    in_place_and_object_grasped = hamacher_product(object_grasped, stick_in_place)

    stick_is_pushing = _stick_is_pushing(stick_pos, handle)

    # X-push: reward for closing the X gap, only after stick is in pushing position
    container_x_to_target_x = torch.abs(handle[:, 0] - target_pos[:, 0])
    x_push_margin = torch.clamp(torch.abs(obj_init_pos[:, 0] - target_pos[:, 0]), min=0.15)
    x_push = tolerance_fixed(container_x_to_target_x, bounds=(0.0, 0.02), margin=x_push_margin, sigmoid="long_tail")
    x_push = torch.where(stick_is_pushing, x_push, torch.zeros_like(x_push))

    # Y-push: reward for closing the Y gap, only after stick is in pushing position
    container_y_to_target_y = torch.abs(handle[:, 1] - target_pos[:, 1])
    y_push_margin = torch.clamp(torch.abs(obj_init_pos[:, 1] - target_pos[:, 1]), min=0.15)
    y_push = tolerance_fixed(container_y_to_target_y, bounds=(0.0, 0.02), margin=y_push_margin, sigmoid="gaussian")
    y_push = torch.where(stick_is_pushing, y_push, torch.zeros_like(y_push))

    rewards = in_place_and_object_grasped
    rewards = torch.where(grasp_success, 1.0 + in_place_and_object_grasped + 5.0 * stick_in_place, rewards)
    rewards = torch.where((grasp_success) & (stick_is_pushing),
                          1.0 + in_place_and_object_grasped + 3.0 + 10.0 * y_push + 5.0 * x_push,
                          rewards) * .01

    success = torch.logical_and((handle_to_target <= .12), torch.logical_and(grasp_success, stick_is_pushing))
    rewards = torch.where(success, 50.0, rewards)

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

    # get obj init pos (thermos body push point: root z 1.027 + 0.17 offset)
    self.obj_init_pos[env_ids] = self.last_rand_vecs[env_ids,:3].clone()
    self.obj_init_pos[env_ids,1] += -.3
    self.obj_init_pos[env_ids,2] = 1.197

    # target is below the stick in the y axis and z is the middle of the handle
    self.target_pos[env_ids] = self.last_rand_vecs[env_ids,3:]

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
    
    # reset thermos DoFs slidex and slidey to 0
    thermos_dof_x_idxs_long = (self.franka_dof_start_idx[env_ids]+self.num_franka_dofs).flatten().to(dtype=torch.long)
    thermos_dof_y_idxs_long = (self.franka_dof_start_idx[env_ids]+self.num_franka_dofs+1).flatten().to(dtype=torch.long)

    all_pos = torch.zeros_like(self.dof_state)
    all_pos[...,0][thermos_dof_x_idxs_long] = torch.zeros_like(self.thermos_dof_upper_limits[0])
    all_pos[...,1][thermos_dof_x_idxs_long] = 0.0
    all_pos[...,0][thermos_dof_y_idxs_long] = torch.zeros_like(self.thermos_dof_upper_limits[1])
    all_pos[...,1][thermos_dof_y_idxs_long] = 0.0

    self.dof_state[...,0].flatten().index_copy_(0,thermos_dof_x_idxs_long.flatten(),all_pos[...,0][thermos_dof_x_idxs_long].flatten())
    self.dof_state[...,1].flatten().index_copy_(0,thermos_dof_x_idxs_long.flatten(),all_pos[...,1][thermos_dof_x_idxs_long].flatten())
    self.dof_state[...,0].flatten().index_copy_(0,thermos_dof_y_idxs_long.flatten(),all_pos[...,0][thermos_dof_y_idxs_long].flatten())
    self.dof_state[...,1].flatten().index_copy_(0,thermos_dof_y_idxs_long.flatten(),all_pos[...,1][thermos_dof_y_idxs_long].flatten())

    # reset thermos and object pose
    thermos_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+1).flatten().to(dtype=torch.int32)
    # not the insertion pos, but the root body pos
    self.root_state_tensor[thermos_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,:3].clone()
    self.root_state_tensor[thermos_multi_env_ids_int32,1] -= .3
    self.root_state_tensor[thermos_multi_env_ids_int32,2] = 1.0270
    self.root_state_tensor[thermos_multi_env_ids_int32,3:7] = to_torch([0, 0, -0.7071068, 0.7071068 ],device=self.device)
    self.root_state_tensor[thermos_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    stick_multi_env_ids_int32 = (self.franka_actor_idx[env_ids]+2).flatten().to(dtype=torch.int32)
    self.root_state_tensor[stick_multi_env_ids_int32,:3] = self.last_rand_vecs[env_ids,:3]
    self.root_state_tensor[stick_multi_env_ids_int32,3:7] = to_torch([0, 0, 0.7071068, 0.7071068 ],device=self.device)
    self.root_state_tensor[stick_multi_env_ids_int32,7:13] = to_torch([0,0,0,0,0,0],device=self.device)

    # Set stick init pos analytically (rigid body states are stale at progress_buf==0)
    sk = self.specialized_kwargs['stick_push']
    sk['stick_init_pos'][env_ids] = self.last_rand_vecs[env_ids,:3].clone()

    actor_multi_env_ids_int32 = torch.cat((thermos_multi_env_ids_int32,stick_multi_env_ids_int32),dim=-1).flatten().to(dtype=torch.int32)

    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0

    dof_multi_env_ids_int32 = torch.cat((self.franka_actor_idx[env_ids],thermos_multi_env_ids_int32),dim=-1).flatten().to(dtype=torch.int32)

    return dof_multi_env_ids_int32,actor_multi_env_ids_int32
