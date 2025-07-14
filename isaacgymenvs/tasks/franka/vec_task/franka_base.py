from collections import defaultdict
from typing import Dict, Tuple
import isaacgym
import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, tensor_clamp
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.franka.vec_task import task_fns

from skrl.utils.isaacgym_utils import ik

TASK_IDX_TO_NAME = {
    0:  "assembly",
    1:  "basketball",
    2:  "bin_picking",
    3:  "box_close",
    4:  "button_press_topdown",
    5:  "button_press_topdown_wall",
    6:  "button_press",
    7:  "button_press_wall",
    8:  "coffee_button",
    9:  "coffee_pull",
    10: "coffee_push",
    11: "dial_turn",
    12: "disassemble",
    13: "door_close",
    14: "door_lock",
    15: "door_unlock",
    16: "door_open",
    17: "drawer_close",
    18: "drawer_open",
    19: "faucet_close",
    20: "faucet_open",
    21: "hammer",
    22: "hand_insert",
    23: "handle_press_side",
    24: "handle_press",
    25: "handle_pull_side",
    26: "handle_pull",
    27: "lever_pull",
    28: "peg_insert_side",
    29: "peg_unplug_side",
    30: "pick_out_of_hole",
    31: "pick_place",
    32: "pick_place_wall",
    33: "plate_slide_back_side",
    34: "plate_slide_back",
    35: "plate_slide_side",
    36: "plate_slide",
    37: "push_back",
    38: "push",
    39: "push_wall",
    40: "reach",
    41: "reach_wall",
    42: "shelf_place",
    43: "soccer",
    44: "stick_pull",
    45: "stick_push",
    46: "sweep_into_goal",
    47: "sweep",
    48: "window_close",
    49: "window_open",
}
FRANKA_DEFAULT_DOF_STATES = [
    [-0.25, .4, 0, -2.2, -0.17, 2.6, -0.7, 0.04, 0.04], # 0
    [0,-.3, 0, -2.8, -.17, 2.6, 0.7, 0.04, 0.04],
    [.15, 0.3, 0, -2, 0, 2.45, 0.7, 0.04, 0.04],
    [0, -.4, 0, -2.9, -0.17, 2.6, -0.7, 0.04, 0.04],
    [0,-.1, 0, -2.6, -.17, 2.6, 0.7, 0.04, 0.04],
    [0,-.1, 0, -2.6, -.17, 2.6, 0.7, 0.04, 0.04],
    [0,-.2, 0, -2.9, -.17, 2.6, 0.7, 0.04, 0.04],
    [0,-.2, 0, -2.9, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, -0.2, 0, -2.7, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, 0.1, 0, -2.5, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, 0.1, 0, -2.5, -.17, 2.6, 0.7, 0.04, 0.04], # 10
    [0, 0.6, 0, -1.7, 0, 2.4, .7, 0.04, 0.04],
    [-0.25, .3, 0, -2.4, -0.17, 2.6, -0.7, 0.04, 0.04],
    [.45, .85, 0.14485, -1.25, -0.2897, 2, -0.1, 0.04, 0.04], # door_close has a different init pos because it must not intersect with the door
    [0, 0, 0, -2.7, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, 0, 0, -2.7, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, 0, 0, -2.7, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, -0.2, 0, -2.9, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, -0.2, 0, -2.9, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.8, 0.7, 0.04, 0.04],  # 20
    [.2, 0.1, 0, -2.6, -0.17, 2.6, -0.7, 0.04, 0.04],
    [0,.2, 0, -2.5, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, -0.3, 0, -2.5, -.17, 2.6, 0, 0.04, 0.04],
    [0, 0.1, 0, -2.5, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, -0.3, 0, -2.5, -.17, 2.6, 0, 0.04, 0.04],
    [0, 0.1, 0, -2.5, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, -0.2, 0, -2.9, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, 0.4, 0, -2.2, 0, 2.5, -0.7, 0.04, 0.04],
    [0,-.1, 0, -2.6, -.17, 2.6, 0.7, 0.04, 0.04], # peg unplug
    [0, 0.75, 0, -1.6, -0.17, 2.5, .8, 0.04, 0.04], # 30
    [0,.2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0,.2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0,.25, 0, -2.6, -.17, 2.8, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.7, 0.7, 0.04, 0.04],
    [0,.25, 0, -2.6, -.17, 2.8, -0.7, 0.04, 0.04],
    [0,.25, 0, -2.6, -.17, 2.8, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04], # 40
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, -.3, 0, -2.8, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, 0.6, 0, -1.45, 0, 2.06, -.87, 0.04, 0.04],
    [0, 0.6, 0, -1.45, 0, 2.06, -.87, 0.04, 0.04], # 45
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.8, 0.7, 0.04, 0.04],
]

# FRANKA_DEFAULT_DOF_STATES = [[0,-.2, 0, -2.9, -.17, 2.6, 0.7, 0.04, 0.04] for _ in range(len(TASK_IDX_TO_NAME))]

def transform_task_indices(task_indices: torch.tensor) -> torch.tensor:
    # if task_indices contains discontinuous integers, transform them to continuous integers
    unique_task_indices = torch.unique(task_indices)
    task_indices_map = {tid.item(): i for i, tid in enumerate(unique_task_indices)}
    return torch.tensor([task_indices_map[tid.item()] for tid in task_indices], device=task_indices.device)

class FrankaBaseEnvV2(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        if self.cfg["env"]["seed"] > 0:
            np.random.seed(self.cfg["env"]["seed"])
            torch.manual_seed(self.cfg["env"]["seed"])

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.reset_noise = self.cfg["env"].get("resetNoise", .1)
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_rendering_interval = self.cfg["env"]["cameraRenderingInterval"]

        self.ml_one_enabled = self.cfg["env"].get("metaLearningEnabled",None)
        self.meta_batch_size = self.cfg["env"].get("metaBatchSize",None)

        self.fixed = self.cfg["env"]["fixed"]
        # is the object position and target fixed per reset?
        self.up_axis = "z"
        self.up_axis_idx = 2

        self.dt = self.cfg['sim']['dt']

        num_obs = 39
        if self.cfg["env"]["taskEmbedding"]:
            num_obs += len(set(self.cfg["env"]["tasks"]))
            
        if self.ml_one_enabled:
            self.num_envs_per_task = self.cfg["env"]["numEnvs"]//self.meta_batch_size
            assert self.cfg["env"]["numEnvs"]%self.meta_batch_size==0, "numEnvs should be divisible by metaBatchSize"
        
        # change in 3D space of the end-effector followed by a normalized torque that the gripper fingers should appl
        num_actions = 4

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_actions

        if "actionSpace" in self.cfg["env"]:
            action_space_type = self.cfg["env"]["actionSpace"]["type"]
            if action_space_type == "multi_discrete":
                self.actions_num = self.cfg["env"]["actionSpace"]["binsPerDim"] # number of bins per dimension
                self.action_dim = num_actions # number of actions
            elif action_space_type == "discrete":
                self.actions_num = self.cfg["env"]["actionSpace"]["numActions"] # number of choices per action dim (which is 1)
                self.action_dim = 1 # number of actions
            else: # continuous
                self.action_dim = num_actions
            
        
        # create tensor placeholders for franka
        self.task_idx = self.cfg["env"]["tasks"]  # list of tasks indexes from 0-49 (50 different possible tasks)
        self.num_tasks = len(self.task_idx)
        self.task_env_count = self.cfg["env"]["taskEnvCount"]  # list of number of envs per task

        self.init_at_random_progress = self.cfg['env']['init_at_random_progress']
        self.termination_on_success = self.cfg["env"]["termination_on_success"]
        self.reward_scale = self.cfg["env"]["reward_scale"]
        self.exempted_init_at_random_progress_tasks = self.cfg["env"]["exemptedInitAtRandomProgressTasks"]
        assert sum(self.task_env_count) == self.cfg["env"]["numEnvs"], \
            f"Sum of taskEnvCount {self.task_env_count} should be equal to num_envs {self.cfg['env']['numEnvs']}"
        assert len(self.task_idx) == len(self.task_env_count), \
            f"Length of task_idx {len(self.task_idx)} should be equal to length of task_env_count {len(self.task_env_count)}"

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim,"franka")

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor)
        self.jacobian = gymtorch.wrap_tensor(jacobian_tensor)
        # torch.Size([num_envs, num_franka_joints (15), 6 spatial degrees of freedoms (3 linear, 3 angular), num_actuated_dofs(9)])

        """
        Seven revolute joints for the arm:
        panda_joint1
        panda_joint2
        panda_joint3
        panda_joint4
        panda_joint5
        panda_joint6
        panda_joint7

        Fixed joints:
        8. panda_hand_joint
        9. panda_hand_y_axis_joint
        10. panda_hand_z_axis_joint
        11. panda_grip_vis_joint
        12. panda_leftfinger_tip_joint
        13. panda_rightfinger_tip_joint

        Prismatic joints for the gripper:
        14. panda_finger_joint1
        15. panda_finger_joint2
        """

        self._init_franka()
        self._init_data()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def compute_reward(self, actions):
        total_count = 0
        env_ids = torch.arange(self.num_envs, device=self.device)
        franka_dof_pos = self.dof_state[self.franka_dof_idx, :].view(self.num_envs, -1, 2)[...,0] # get franka dof pos
        for tid, env_count in zip(self.task_idx, self.task_env_count):
            task_name = self.task_idx2name[tid]
            reward_fn = getattr(task_fns, task_name).compute_reward
            ids = env_ids[total_count:total_count+env_count]
            self.rew_buf[ids], self.reset_buf[ids], self.success_buf[ids] = reward_fn(
                self.reset_buf[ids],self.progress_buf[ids], self.actions[ids], franka_dof_pos[ids],
                self.franka_lfinger_pos[ids], self.franka_rfinger_pos[ids], self.max_episode_length,
                self.tcp_init[ids], self.target_pos[ids], self.obj_pos[ids], self.obj_init_pos[ids], self.specialized_kwargs[task_name][ids[0].item()])
            if self.cfg["env"]["sparse_reward"]:
                self.rew_buf[ids] = self.success_buf[ids].float()

            total_count += env_count

        if not self.termination_on_success:
            self.reset_buf = (self.progress_buf >= self.max_episode_length - 1)

        scaling_factor = torch.ones_like(self.rew_buf) * self.reward_scale
        scaling_factor[self.success_buf] = 1.0
        self.rew_buf *= scaling_factor

    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        
        # ------------------- Universal observation for all tasks -------------------#
        # universal observation: end effector pos and openness (3 + 1)

        panda_leftfinger_tip_rigid_body_idx = self.franka_rigid_body_start_idx[:] + 10
        panda_rightfinger_tip_rigid_body_idx = self.franka_rigid_body_start_idx[:] + 12
        panda_eef_rigid_body_idx = self.franka_rigid_body_start_idx[:] + 13 # panda_grip_site

        panda_leftfinger_tip_rigid_body_states = self.rigid_body_states[panda_leftfinger_tip_rigid_body_idx].view(-1, 13)
        panda_rightfinger_tip_rigid_body_states = self.rigid_body_states[panda_rightfinger_tip_rigid_body_idx].view(-1, 13)
        panda_eef_rigid_body_states = self.rigid_body_states[panda_eef_rigid_body_idx].view(-1, 13)
        
        eef_pos = panda_eef_rigid_body_states[:, 0:3]
        eef_rot = panda_eef_rigid_body_states[:, 3:7]
        
        self.franka_lfinger_pos = panda_leftfinger_tip_rigid_body_states[:, 0:3]
        self.franka_rfinger_pos = panda_rightfinger_tip_rigid_body_states[:, 0:3]
        
        gripper_distance_apart = torch.norm(
            self.franka_rfinger_pos - self.franka_lfinger_pos, dim=-1)
        normalized_openess = torch.clip(
            gripper_distance_apart/.095, 0.0, 1.0).unsqueeze(-1)
        
        # get init tcp
        if self.tcp_init is None:     
            self.tcp_init = (self.franka_lfinger_pos + self.franka_rfinger_pos) / 2
        
        # ------------------- Task specific observation -------------------#
        # task specific observation: object1 pos rot and object2 pos rot (3 + 4 + 3 + 4)
        total_count = 0
        env_ids = torch.arange(self.num_envs, device=self.device)
        object_states = []
        for tid, env_count in zip(self.task_idx, self.task_env_count):
            task_name = self.task_idx2name[tid]
            obs_fn = getattr(task_fns, task_name).compute_observations
            object_state = obs_fn(self, env_ids[total_count:total_count+env_count])
            total_count += env_count
            
            object_states.append(object_state)
        object_states = torch.cat(object_states, dim=0)
        
        self.obj_pos = object_states[:, 0:3]

        # ------------------- add task specific observation + goal -------------------#
        if self.ml_one_enabled:
            self.obs_buf = torch.cat([
                eef_pos,
                normalized_openess, 
                object_states,
                self.obs_buf[:, 0:18],
                torch.zeros_like(self.target_pos)
            ], dim=-1)
        else:
            self.obs_buf = torch.cat([
                eef_pos,
                normalized_openess, 
                object_states,
                self.obs_buf[:, 0:18],
                self.target_pos
            ], dim=-1)

        if self.cfg["env"]["taskEmbedding"]:
            self.obs_buf = torch.cat([
                self.obs_buf,
                self.task_embedding
            ], dim=-1)
    

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        
        # Assuming actions are [dx, dy, dz, normalized_torque]
        u_eef_delta, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        panda_eef_rigid_body_idx = self.franka_rigid_body_start_idx[:] + 13 # panda_grip_site
        panda_eef_rigid_body_states = self.rigid_body_states[panda_eef_rigid_body_idx].view(-1, 13)
        eef_pos = panda_eef_rigid_body_states[:, 0:3]
        eef_rot = panda_eef_rigid_body_states[:, 3:7]

        goal_position = eef_pos + u_eef_delta * self.action_scale
        j_eef = self.jacobian[:, 7, :, :7] # panda_hand_joint connects last link to panda hand

        # Calculate joint deltas using IK
        dof_deltas = ik(
            jacobian_end_effector=j_eef,
            current_position=eef_pos,
            current_orientation=eef_rot,
            goal_position=goal_position,
            damping_factor=0.05
        )

        # Add velocity damping
        # velocity_damping = 0.1  # Tune this parameter
        # dof_delta_magnitude = torch.norm(dof_deltas, dim=1, keepdim=True)
        # velocity_scaling = torch.exp(-velocity_damping * dof_delta_magnitude)
        # dof_deltas = dof_deltas * velocity_scaling

        # Calculate new joint targets
        cur_targets = self.dof_targets_all[self.franka_dof_idx].view(self.num_envs,-1)[:, :-2]
        targets = cur_targets + dof_deltas

        # Apply joint limits
        targets = tensor_clamp(
            targets,
            self.franka_dof_lower_limits[:-2],
            self.franka_dof_upper_limits[:-2]
        )
        
        franka_dof_pos_idx = self.franka_dof_idx.view(self.num_envs, self.num_franka_dofs)[:, :-2].to(dtype=torch.long)
        self.dof_targets_all.flatten().index_copy_(0, franka_dof_pos_idx.flatten(), targets.flatten())
        
        # Clip the values to be within valid effort range
        u_fingers = torch.zeros_like(self._gripper_control)
        u_gripper *= 100 # Map from [-1, 1] to [-100, 100]

        u_fingers[:, 0] = torch.clamp(u_gripper, -self._franka_effort_limits[-2].item(),
                                      self._franka_effort_limits[-2].item())
        u_fingers[:, 1] = torch.clamp(u_gripper, -self._franka_effort_limits[-1].item(),
                                      self._franka_effort_limits[-1].item())
        
        
        # Write gripper command to effort_control_all
        self._gripper_control[:, :] = u_fingers
        franka_effort_idx = self.franka_dof_idx.view(self.num_envs, self.num_franka_dofs)[:,-2:].to(dtype=torch.long)
        self.effort_control_all.flatten().index_copy_(0,franka_effort_idx.flatten(), self._gripper_control.flatten())
        
        # Deploy actions
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_targets_all),
            gymtorch.unwrap_tensor(self.franka_actor_idx),
            len(self.franka_actor_idx)
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.effort_control_all),
            gymtorch.unwrap_tensor(self.franka_actor_idx),
            len(self.franka_actor_idx)
        )

    def render_headless(self):
        if self.control_steps % 5 == 0:
            # imageio.imwrite(f'debug/scene_{self.control_steps}.png', self.torch_camera_tensor.cpu().numpy())
            self.camera_frames.append(self.torch_camera_tensor.clone())

    def get_debug_viz(self):
        if not self.debug_viz or not self.camera_request_visual:
            return None

        if self.camera_request_visual:
            if len(self.camera_frames) == 0:
                if self.progress_buf[-1].item() == 1:
                    # only start rendering at first time step
                    self.camera_rendering = True
                return None
        
            if len(self.camera_frames) < self.max_episode_length / 5:
                return None
            else:
                self.camera_rendering = False
                self.camera_request_visual = False
                camera_frames = torch.stack(self.camera_frames, dim=0).permute(0, 3, 1, 2).unsqueeze(0)
                self.camera_frames = []
                return camera_frames
        else:
            return None

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.camera_rendering:
            self.render_headless()

        if self.control_steps % self.camera_rendering_interval == 0:
            self.camera_request_visual = True

        # update self.extras (not neccessay they are always the same)
        # self.extras["task_indices"] = transform_task_indices(self.task_indices)
        # self.extras["episode_cumulative"]["success_at_end"] = self.success_buf.clone()
        
        self.cumulatives["reward"] += self.rew_buf
        self.cumulatives["success"] += self.success_buf
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        
        # reward, success, and episode length logging
        if len(env_ids) > 0:
            self.extras['episode'] = {}
            task_success_rates = []

            # among the environments that just finished an episode, what proportion of them achieved success at least once during their episode?
            average_environment_success_rate = (self.cumulatives["success"][env_ids].clone() > 0).clone().float().mean().item()
            self.extras['episode']["average_environment_success_rate"] = average_environment_success_rate
            self.extras['episode']['success'] = self.cumulatives["success"].clone() > 0
            # among the environments that just finished an episode, what is normalized average number of successes per episode?
            self.extras['episode']["success_count_per_episode"] = self.cumulatives["success"][env_ids].clone()

            for i,tid in enumerate(self.task_idx):
                # Get environments that finished for this task
                env_ids_counted = torch.logical_and(self.task_indices == tid, self.reset_buf).nonzero(as_tuple=False).squeeze(-1)

                if len(env_ids_counted) > 0:
                    task_success = (self.cumulatives["success"][env_ids_counted].clone() > 0).clone().float().mean().item()
                    task_success_rates.append(task_success)
                
                # among the environments for this task that just finished an episode, what is the cumulative reward for each of them?
                self.extras['episode']["task_{}_reward".format(tid)] = self.cumulatives["reward"][env_ids_counted].clone()
                # among the environments for this task that just finished an episode, what proportion of them achieved success at least once during their episode?
                self.extras['episode']["task_{}_success".format(tid)] =  self.cumulatives["success"][env_ids_counted].clone() > 0
                self.extras['episode']["task_{}_eplength".format(tid)] = self.progress_buf[env_ids_counted].clone()
                # among the environments for this task that just finished an episode, what is the normalized average number of successes per episode?
                self.extras['episode']["task_{}_success_count_per_episode".format(tid)] = self.cumulatives["success"][env_ids_counted].clone()

            # Calculate average success rate across tasks (only for tasks that had finished episodes)
            if task_success_rates:
                average_task_success = sum(task_success_rates) / len(task_success_rates)
                self.extras['episode']["average_task_success_rate"] = average_task_success

            self.cumulatives["reward"][env_ids] = 0
            self.cumulatives["success"][env_ids] = 0

        if self.debug_viz and self.camera_request_visual:
            self.extras["debug_visual"] = self.get_debug_viz()
        else:
            self.extras["debug_visual"] = None
    
    @property
    def task_idx2name(self):
        return TASK_IDX_TO_NAME
    
    def _init_franka(self):
        """Initialize values for the franka. For examples, 
        the default DoF position, velocity, and target tensors.
        """
        self.franka_default_dof_pos = [to_torch(FRANKA_DEFAULT_DOF_STATES[i],device=self.device) for i in range(len(FRANKA_DEFAULT_DOF_STATES))]
        self.franka_dof_idx = torch.stack([self.franka_dof_start_idx + i for i in range(self.num_franka_dofs)], dim=1).flatten()
        self.franka_dof_states = self.dof_state[self.franka_dof_idx, :].view(self.num_envs, -1, 2) # creates a copy!
        self.franka_dof_pos = self.franka_dof_states[..., 0]
        self.franka_dof_vel = self.franka_dof_states[..., 1]
        
        franka_rigid_body_idx = torch.stack([self.franka_rigid_body_start_idx + i for i in range(self.num_franka_rigid_bodies)], dim=1).flatten()
        self.franka_rigid_body_states = self.rigid_body_states[franka_rigid_body_idx, :].view(self.num_envs, -1, 13) # creates a copy!
                
        self.dof_targets_all = torch.zeros((self.gym.get_sim_dof_count(self.sim), 1), device=self.device).to(dtype=torch.float)
        self.franka_dof_targets = self.dof_targets_all[self.franka_dof_idx].view(self.num_envs, -1) # creates a copy!

        self.effort_control_all = torch.zeros((self.gym.get_sim_dof_count(self.sim), 1), dtype=torch.float, device=self.device)
        self._effort_control = self.effort_control_all[self.franka_dof_idx].view(self.num_envs, -1) # creates a copy!
        self._gripper_control = self._effort_control[:, 7:9]

        
        self.task_indices = []
        for i, count in zip(self.task_idx, self.task_env_count):
            self.task_indices.extend([i]*count)
        self.task_indices = torch.tensor(self.task_indices, device=self.device)

        if self.cfg["env"]["taskEmbedding"]:
            # Transform task indices to get continuous indices
            transformed_indices = transform_task_indices(self.task_indices)
            num_unique_tasks = len(torch.unique(transformed_indices))
            
            # Use the transformed indices for one-hot encoding
            self.task_embedding = torch.nn.functional.one_hot(
                transformed_indices, 
                num_unique_tasks
            ).float()
        self.success_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        self.extras["episode_cumulative"] = {}
        # transform task indices to be consecutive integers to reduce task embedding dim 
        self.extras["task_indices"] = transform_task_indices(self.task_indices)
        # record task names
        ordered_task_names = [self.task_idx2name[tid.item()] for tid in torch.unique(self.task_indices)]
        self.extras["ordered_task_names"] = list(map(lambda s: s.replace("_", "-"), ordered_task_names))


        self.cumulatives = defaultdict(lambda: torch.zeros(self.num_envs, device=self.device))

        if self.debug_viz:
            self.camera_frames = []

        self.camera_rendering = False  # by default, don't render camera except for viewer camera, which is controlled by headless flag
        self.camera_request_visual = False
        
    def _create_envs(self, num_envs, spacing, num_per_row):
        #------------------- prepare Franka and table assets shared by all envs -------------------#
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"
        
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
        
        # load franka asset
        franka_asset_options = gymapi.AssetOptions()
        franka_asset_options.flip_visual_attachments = True
        franka_asset_options.fix_base_link = True
        franka_asset_options.collapse_fixed_joints = False
        franka_asset_options.disable_gravity = True
        franka_asset_options.thickness = 0.001
        franka_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        franka_asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, franka_asset_options)
        
        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.054
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)
        
        # Create table stand asset
        table_stand_height = 0.01
        table_stand_pos = [-0.6, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)
        
        franka_dof_stiffness = to_torch([1600, 1600, 1600, 1600, 1600, 1600, 1600, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.num_franka_rigid_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i <= 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                if i <= 6:
                    franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                    franka_dof_props['damping'][i] = franka_dof_damping[i]
                else:
                    franka_dof_props['stiffness'][i] = 0
                    franka_dof_props['damping'][i] = 0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self._franka_effort_limits[[7,8]] = 50
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        # franka_dof_props['effort'][7] = 100
        # franka_dof_props['effort'][8] = 100
        
        # define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.55, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        self.num_table_rigid_bodies = self.gym.get_asset_rigid_body_count(table_asset) \
            + self.gym.get_asset_rigid_body_count(table_stand_asset)
        #-------------------------------------------------------------------------------------#
        
        self.frankas = []
        self.envs = []
        self.objects = []
        self.random_reset_space_task = [None]*len(FRANKA_DEFAULT_DOF_STATES)
        self.franka_actor_idx = torch.zeros(self.num_envs, device=self.device).to(dtype=torch.int32)
        # where the first franka dof starts in each env 
        self.franka_dof_start_idx = torch.zeros(self.num_envs, device=self.device).to(dtype=torch.int32)
        # where the first franka rigid body starts in each env 
        self.franka_rigid_body_start_idx = torch.zeros(self.num_envs, device=self.device).to(dtype=torch.int32)
        i = 0  # count the current env idx
        
        total_actor_count = 0
        total_dof_count = 0
        total_rigid_body_count = 0
        for tid, env_count in zip(self.task_idx, self.task_env_count):
            task_name = self.task_idx2name[tid]
            print("Creating %d Franka environments for task %s" % (env_count, task_name))
            # create envs
            env_fn = getattr(task_fns, task_name).create_envs
            envs, franks, objects, random_reset_space, num_task_actors, num_task_dof, num_task_rigid_bodies = env_fn(
                self, env_count, spacing, num_per_row,
                franka_asset, franka_start_pose, franka_dof_props,
                table_asset, table_start_pose, table_stand_asset, table_stand_start_pose,    
            )
            self.envs.extend(envs)
            self.frankas.extend(franks)
            self.objects.extend(objects)
            self.random_reset_space_task[tid] = random_reset_space
            for _ in range(env_count):
                self.franka_actor_idx[i] = total_actor_count
                total_actor_count += num_task_actors + 3
                
                self.franka_dof_start_idx[i] = total_dof_count
                total_dof_count += num_task_dof + self.num_franka_dofs
                
                self.franka_rigid_body_start_idx[i] = total_rigid_body_count
                total_rigid_body_count += \
                    num_task_rigid_bodies + self.num_franka_rigid_bodies + self.num_table_rigid_bodies
                
                i += 1
            
        env_ptr = self.envs[0]
        franka_actor = self.frankas[0]
        self.eef_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_grip_site")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger_tip")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger_tip")
        
        # add camera
        # TODO: need to add camera to each env for visual based tasks
        if self.debug_viz:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = self.cfg["env"]["cameraWidth"]
            self.camera_props.height = self.cfg["env"]["cameraHeight"]
            self.camera_props.enable_tensors = True
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[-1], self.camera_props)
            if self.rendering_camera!=-1:
                self.gym.set_camera_location(self.rendering_camera, self.envs[-1], gymapi.Vec3(-.5, -1, 1.5), gymapi.Vec3(0, 0, 1))

                # obtain camera tensors
                video_frame_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[-1], self.rendering_camera, gymapi.IMAGE_COLOR)
                self.torch_camera_tensor = gymtorch.wrap_tensor(video_frame_tensor).view((self.camera_props.height, self.camera_props.width, 4))

            img_dir = "debug"
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
    
    def _init_data(self):
        self.tcp_init = None
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.specialized_kwargs = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(dict)
            )
        )
        self.env_id_to_task_id = [-1 for _ in range(self.num_envs)]

        self.last_rand_vecs = torch.zeros((self.num_envs,6),device=self.device)
        self.last_rand_vecs_test = torch.zeros((self.num_envs,6),device=self.device)

        total_count = 0
        for tid, env_count in zip(self.task_idx, self.task_env_count):
            task_name = self.task_idx2name[tid]
            self.specialized_kwargs[task_name][total_count] = {}
            if task_name == 'basketball':
                self.specialized_kwargs[task_name][total_count]['scale'] = to_torch([1.0, 1.0, 1.0], device=self.device)
            elif task_name == 'box_close':
                self.specialized_kwargs[task_name][total_count]['error_scale'] = to_torch([1.0, 1.0, 3.0], device=self.device)
            elif task_name == 'coffee_pull':
                self.specialized_kwargs[task_name][total_count]['scale'] = to_torch([2.0, 2.0, 1.0],device=self.device)
            elif task_name == 'coffee_push':
                self.specialized_kwargs[task_name][total_count]['scale'] = to_torch([2.0, 2.0, 1.0],device=self.device)
            elif task_name == 'dial_turn':
                self.specialized_kwargs[task_name][total_count]['dial_push_position_init'] = None
            elif task_name == 'door_lock': 
                self.specialized_kwargs[task_name][total_count]['scale'] = to_torch([1,.25,.5], device=self.device)
            elif task_name == 'door_unlock':
                self.specialized_kwargs[task_name][total_count]['scale'] = to_torch([1.0, 1.0, 1.0], device=self.device)
            elif task_name == 'drawer_open':
                self.specialized_kwargs[task_name][total_count]['scale'] = to_torch([1.0, 1.0, 1.0],device=self.device)
            elif task_name == 'handle_pull_side':
                self.specialized_kwargs[task_name][total_count]['scale'] = to_torch([1.0, 1.0, 1.0],device=self.device)
            elif task_name == 'lever_pull':
                self.specialized_kwargs[task_name][total_count]['scale'] =  to_torch([1.0,4.0,4.0],device=self.device)
                self.specialized_kwargs[task_name][total_count]['offset'] = to_torch([0, 0, 0.07],device=self.device)
                self.specialized_kwargs[task_name][total_count]['lever_pos_init'] = None
            elif task_name == 'peg_insert_side':
                self.specialized_kwargs[task_name][total_count]['peg_head_pos_init'] = None
                self.specialized_kwargs[task_name][total_count]['scale'] = to_torch([2.0, 1.0, 2.0], device=self.device)
            elif task_name == 'pick_place_wall':
                self.specialized_kwargs[task_name][total_count]['in_place_scaling'] = to_torch([3.0, 1.0, 1.0], device=self.device)
            elif task_name == 'push_wall':
                self.specialized_kwargs[task_name][total_count]['in_place_scaling'] = to_torch([3.0, 1.0, 1.0], device=self.device)
                self.specialized_kwargs[task_name][total_count]['midpoint'] = to_torch([0.17, .1, 1.0470], device=self.device)
            elif task_name == 'soccer':
                self.specialized_kwargs[task_name][total_count]['scale'] = to_torch([1.0, 3.0, 1.0],device=self.device)
            elif task_name == 'stick_push':
                self.specialized_kwargs[task_name][total_count]['stick_init_pos'] = None
                self.specialized_kwargs[task_name][total_count]['thermos_dof_pos'] = None
            elif task_name == 'stick_pull':
                self.specialized_kwargs[task_name][total_count]['stick_init_pos'] = None
                self.specialized_kwargs[task_name][total_count]['yz_scaling'] = to_torch([1.0, 1.0, 2.0], device=self.device)
                self.specialized_kwargs[task_name][total_count]['thermos_dof_pos'] = None
                self.specialized_kwargs[task_name][total_count]['thermos_insertion_pos'] = None
                self.specialized_kwargs[task_name][total_count]['thermos_insertion_pos_init'] = None
            elif task_name == 'sweep':
                self.specialized_kwargs[task_name][total_count]['init_left_pad'] = None
                self.specialized_kwargs[task_name][total_count]['init_right_pad'] = None
                self.specialized_kwargs[task_name][total_count]['scale'] = to_torch([1.0, 1.0, 1.0],device=self.device)
            total_count += env_count
            
        total_count = 0
        for tid, env_count in zip(self.task_idx, self.task_env_count):
            self.env_id_to_task_id[total_count:total_count+env_count] = [tid for _ in range(env_count)]
            # exempted_init_at_random_progress_tasks is a list of task ids that should not be initialized at random progress
            if not self.init_at_random_progress or tid in self.exempted_init_at_random_progress_tasks:
                self.progress_buf[total_count:total_count+env_count] = 0
            else:
                self.progress_buf[total_count:total_count+env_count] = torch.randint_like(self.progress_buf[total_count:total_count+env_count], high=int(self.max_episode_length))
                
            total_count += env_count

        #------------------- initialize special variables required for some reward functions -------------------#
        # self.specialized_kwargs['basketball']['scale'] = to_torch([1.0, 1.0, 1.0], device=self.device)
        
        # self.specialized_kwargs['box_close']['error_scale'] = to_torch([1.0, 1.0, 3.0], device=self.device)
        
        # self.specialized_kwargs['coffee_pull']['scale'] = to_torch([2.0, 2.0, 1.0],device=self.device)
        
        # self.specialized_kwargs['coffee_push']['scale'] = to_torch([2.0, 2.0, 1.0],device=self.device)
        
        # self.specialized_kwargs['dial_turn']['dial_push_position_init'] = None
        
        # self.specialized_kwargs['door_lock']['scale'] = to_torch([1,.25,.5], device=self.device)
        
        # self.specialized_kwargs['door_unlock']['scale'] = to_torch([1.0, 1.0, 1.0], device=self.device)

        # self.specialized_kwargs['drawer_open']['scale'] = to_torch([1.0, 1.0, 1.0],device=self.device)

        # self.specialized_kwargs['handle_pull_side']['scale'] = to_torch([1.0, 1.0, 1.0],device=self.device)

        # self.specialized_kwargs['lever_pull']['scale'] =  to_torch([1.0,4.0,4.0],device=self.device)
        # self.specialized_kwargs['lever_pull']['offset'] = to_torch([0, 0, 0.07],device=self.device)
        # self.specialized_kwargs['lever_pull']['lever_pos_init'] = None

        # self.specialized_kwargs['peg_insert_side']['peg_head_pos_init'] = None
        # self.specialized_kwargs['peg_insert_side']['scale'] = to_torch([2.0, 1.0, 2.0], device=self.device)
        
        # self.specialized_kwargs['pick_place_wall']['in_place_scaling'] = to_torch([3.0, 1.0, 1.0], device=self.device)

        # self.specialized_kwargs["push_wall"]["in_place_scaling"] = to_torch([3.0, 1.0, 1.0], device=self.device)
        # self.specialized_kwargs["push_wall"]["midpoint"] = to_torch([0.17, .1, 1.0470], device=self.device)

        # self.specialized_kwargs['soccer']['scale'] = to_torch([1.0, 3.0, 1.0],device=self.device)
        
        # self.specialized_kwargs['stick_push']['stick_init_pos'] = None
        # self.specialized_kwargs['stick_push']['thermos_dof_pos'] = None

        # self.specialized_kwargs['stick_pull']['stick_init_pos'] = None
        # self.specialized_kwargs['stick_pull']['yz_scaling'] = to_torch([1.0, 1.0, 2.0], device=self.device)
        # self.specialized_kwargs['stick_pull']['thermos_dof_pos'] = None
        # self.specialized_kwargs['stick_pull']['thermos_insertion_pos'] = None
        # self.specialized_kwargs['stick_pull']['thermos_insertion_pos_init'] = None

        # self.specialized_kwargs['sweep']['init_left_pad'] = None
        # self.specialized_kwargs['sweep']['init_right_pad'] = None
        # self.specialized_kwargs['sweep']['scale'] = to_torch([1.0, 1.0, 1.0],device=self.device)

    def reset_idx(self, env_ids):
        # map tid -> env_ids
        tids = {}
        # aggregate env_ids by task id
        for env_id in env_ids:
            tid = self.env_id_to_task_id[env_id]
            if tid not in tids:
                tids[tid] = []
            tids[tid].append(env_id)

        dof_multi_env_ids_int32 = []
        actor_multi_env_ids_int32 = []
        # reset each task's envs
        for tid in tids:
            task_name = self.task_idx2name[tid]
            ids = torch.stack(tids[tid])
            reset_fn = getattr(task_fns, task_name).reset_env
            # return the actor indices whose dof needs to be reset and actor indices whose pose needs to be reset
            reset_dof_actor_indices, reset_root_actor_indices = reset_fn(self, tid, ids, self.random_reset_space_task[tid])

            dof_multi_env_ids_int32.append(reset_dof_actor_indices)
            actor_multi_env_ids_int32.append(reset_root_actor_indices)

        dof_multi_env_ids_int32 = torch.hstack(dof_multi_env_ids_int32).flatten().to(dtype=torch.int32)
        actor_multi_env_ids_int32 = torch.hstack(actor_multi_env_ids_int32).flatten().to(dtype=torch.int32)
        franka_env_ids_int32 = self.franka_actor_idx[env_ids].to(dtype=torch.int32)
                
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.dof_targets_all),
                                                        gymtorch.unwrap_tensor(franka_env_ids_int32), 
                                                        len(franka_env_ids_int32))
        
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.effort_control_all),
                                                        gymtorch.unwrap_tensor(franka_env_ids_int32),
                                                        len(franka_env_ids_int32))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                        gymtorch.unwrap_tensor(self.root_state_tensor),
                                                        gymtorch.unwrap_tensor(actor_multi_env_ids_int32), 
                                                        len(actor_multi_env_ids_int32))
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(dof_multi_env_ids_int32), 
                                              len(dof_multi_env_ids_int32))
        
        self.success_buf[env_ids] = 0
        self.extras.pop('episode', None)

    def set_meta_parameters(self,train, num_total_tasks):
        """ Randomly sample meta_batch_size tasks from the set of all 50 tasks for either training or testing,
            and set the last_rand_vecs to the sampled training/testing tasks.
        MUST call reset after this function is called to reset the envs to the new tasks.

        Returns the sampled tasks as indexes
        """
        assert self.meta_batch_size <= num_total_tasks, "meta_batch_size should be less than or equal to num_total_tasks"

        task_indices = np.random.permutation(num_total_tasks)[:self.meta_batch_size] # permute [0,...,49]
        # task_indices = np.arange(num_total_tasks)[:self.meta_batch_size]
        if train:
            for i, task_idx in enumerate(task_indices):
                self.last_rand_vecs[i*self.num_envs_per_task:(i+1)*self.num_envs_per_task] = self.all_train_tasks[task_idx]
            return task_indices
        else:
            self.last_rand_vecs[:] = self.all_train_tasks[task_indices[0]]
            return [task_indices[0]]

    ##########################################################################################