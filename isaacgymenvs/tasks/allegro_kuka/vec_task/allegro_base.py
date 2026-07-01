# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Multi-task Allegro-Kuka environment for MTBench.

This is the Allegro-Kuka analogue of ``franka/vec_task/franka_base.py``:
a single :class:`VecTask` that runs several Allegro-Kuka manipulation tasks at
once.  Each task occupies a contiguous block of environments, all sharing the
same Kuka iiwa7 arm + Allegro hand robot, the same manipulated cube, and the
same table.  Tasks differ in the goal representation / extra object
(regrasping -> ball goal, throw -> bucket, reorientation -> goal cube), in how
the target is sampled (dispatched through the ``task_fns`` package), and in a
set of per-task parameters (episode length, success steps, applied forces,
success tolerance + curriculum, and the procedurally generated object set) that
mirror the per-subtask configs used in SAPG.

The heavy machinery (robot loading, ``full_state`` observation,
``compute_kuka_reward``, reset/physics loop) is lifted from the standalone
``allegro_kuka/allegro_kuka_base.py`` in SAPG; the multi-task scaffolding (task
blocks, per-task dispatch, per-task logging, task embedding) mirrors
``FrankaBaseEnvV2``.

Only single-arm tasks are supported here (23 actions).  The two-arm
reorientation task uses a different robot (46 actions) and is intentionally not
part of this set.
"""

import os
import tempfile
from collections import defaultdict
from copy import copy
from os.path import join
from typing import List, Tuple

import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil
from torch import Tensor

from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_utils import (
    DofParameters,
    populate_dof_properties,
    tolerance_curriculum,
    tolerance_successes_objective,
)
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.allegro_kuka.generate_cuboids import (
    generate_big_cuboids,
    generate_default_cube,
    generate_small_cuboids,
    generate_sticks,
)
from isaacgymenvs.tasks.allegro_kuka.vec_task import task_fns
from isaacgymenvs.utils.torch_jit_utils import *  # noqa: F401,F403  (to_torch, quat_rotate, scale, unscale, ...)


TASK_IDX_TO_NAME = {
    0: "regrasping",
    1: "throw",
    2: "reorientation",
}

# parameters that SAPG overrides per subtask; everything else is shared
PER_TASK_PARAM_KEYS = [
    "episodeLength",
    "successSteps",
    "forceScale",
    "successTolerance",
    "targetSuccessTolerance",
    "withSmallCuboids",
    "withBigCuboids",
    "withSticks",
]


def transform_task_indices(task_indices: torch.Tensor) -> torch.Tensor:
    # if task_indices contains discontinuous integers, transform them to continuous integers
    unique_task_indices = torch.unique(task_indices)
    task_indices_map = {tid.item(): i for i, tid in enumerate(unique_task_indices)}
    return torch.tensor([task_indices_map[tid.item()] for tid in task_indices], device=task_indices.device)


class AllegroKukaBaseEnv(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.frame_since_restart: int = 0  # number of control steps since last restart across all actors

        self.hand_arm_asset_file: str = self.cfg["env"]["asset"]["kukaAllegro"]

        self.clamp_abs_observations: float = self.cfg["env"]["clampAbsObservations"]

        self.privileged_actions = self.cfg["env"]["privilegedActions"]
        self.privileged_actions_torque = self.cfg["env"]["privilegedActionsTorque"]

        # 4 joints for index, middle, ring, and thumb and 7 for kuka arm
        self.num_arm_dofs = 7
        self.num_finger_dofs = 4
        self.num_allegro_fingertips = 4
        self.num_hand_dofs = self.num_finger_dofs * self.num_allegro_fingertips
        self.num_hand_arm_dofs = self.num_hand_dofs + self.num_arm_dofs

        self.num_allegro_kuka_actions = self.num_hand_arm_dofs
        if self.privileged_actions:
            self.num_allegro_kuka_actions += 3

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.distance_delta_rew_scale = self.cfg["env"]["distanceDeltaRewScale"]
        self.lifting_rew_scale = self.cfg["env"]["liftingRewScale"]
        self.lifting_bonus = self.cfg["env"]["liftingBonus"]
        self.lifting_bonus_threshold = self.cfg["env"]["liftingBonusThreshold"]
        self.keypoint_rew_scale = self.cfg["env"]["keypointRewScale"]
        self.kuka_actions_penalty_scale = self.cfg["env"]["kukaActionsPenaltyScale"]
        self.allegro_actions_penalty_scale = self.cfg["env"]["allegroActionsPenaltyScale"]

        self.dof_params: DofParameters = DofParameters.from_cfg(self.cfg)

        # shared (non per-task) tolerance-curriculum knobs
        self.tolerance_curriculum_increment = self.cfg["env"]["toleranceCurriculumIncrement"]
        self.tolerance_curriculum_interval = self.cfg["env"]["toleranceCurriculumInterval"]

        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]

        self.reset_position_noise_x = self.cfg["env"]["resetPositionNoiseX"]
        self.reset_position_noise_y = self.cfg["env"]["resetPositionNoiseY"]
        self.reset_position_noise_z = self.cfg["env"]["resetPositionNoiseZ"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise_fingers = self.cfg["env"]["resetDofPosRandomIntervalFingers"]
        self.reset_dof_pos_noise_arm = self.cfg["env"]["resetDofPosRandomIntervalArm"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]

        # 1.0 means keypoints correspond to the corners of the object
        self.keypoint_scale = self.cfg["env"]["keypointScale"]
        # size of the object (i.e. cube) before scaling
        self.object_base_size = self.cfg["env"]["objectBaseSize"]
        # whether to sample random object dimensions
        self.randomize_object_dimensions = self.cfg["env"]["randomizeObjectDimensions"]

        self.with_dof_force_sensors = False
        self.with_fingertip_force_sensors = False

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block"]

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",  # 0.05m box
            "table": "urdf/table_narrow.urdf",
            "bucket": "urdf/objects/bucket.urdf",
            "ball": "urdf/objects/ball.urdf",
        }

        # ------------------- Multi-task setup (mirrors FrankaBaseEnvV2) -------------------#
        self.task_idx = self.cfg["env"]["tasks"]
        self.num_tasks = len(self.task_idx)
        self.task_env_count = self.cfg["env"]["taskEnvCount"]
        for tid in self.task_idx:
            assert tid in TASK_IDX_TO_NAME, f"Unknown allegro-kuka task id {tid}. Known: {TASK_IDX_TO_NAME}"
        assert sum(self.task_env_count) == self.cfg["env"]["numEnvs"], (
            f"Sum of taskEnvCount {self.task_env_count} should be equal to num_envs {self.cfg['env']['numEnvs']}"
        )
        assert len(self.task_idx) == len(self.task_env_count), (
            f"Length of task_idx {len(self.task_idx)} should equal length of task_env_count {len(self.task_env_count)}"
        )
        self.unique_task_ids = list(dict.fromkeys(self.task_idx))  # preserves order, dedupes

        # per-env task id as a python list (needed inside _create_envs, which runs during
        # super().__init__(); the gpu tensor version is built afterwards)
        self.env_id_to_task_id = []
        for tid, count in zip(self.task_idx, self.task_env_count):
            self.env_id_to_task_id.extend([tid] * count)

        # resolve per-task parameters (SAPG subtask overrides on top of env-level defaults)
        self.task_params = self._resolve_task_params()

        # scalars derived from the per-task values for code paths that need one number
        # (e.g. the MT agent's eval loop does range(env.max_episode_length))
        self.max_episode_length = max(int(self.task_params[t]["episodeLength"]) for t in self.unique_task_ids)
        self.success_steps = max(int(self.task_params[t]["successSteps"]) for t in self.unique_task_ids)
        self.force_scale_max = max(float(self.task_params[t]["forceScale"]) for t in self.unique_task_ids)
        # initial/target tolerance scalars (per-task copies live in dicts built after super().__init__)
        self.initial_tolerance = max(float(self.task_params[t]["successTolerance"]) for t in self.unique_task_ids)
        self.target_tolerance = min(float(self.task_params[t]["targetSuccessTolerance"]) for t in self.unique_task_ids)
        self.success_tolerance = self.initial_tolerance

        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (self.control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        # The full_state observation includes per-keypoint goal info. Different tasks use a
        # different number of keypoints (regrasping/throw -> 1, reorientation -> 4). To keep a
        # single shared observation/keypoint tensor across all env blocks we size everything to
        # the maximum number of keypoints over the selected tasks and pad smaller tasks by
        # replicating their (single, centered) keypoint -- this preserves the orientation-agnostic
        # reward semantics of regrasping/throw.
        self.task_keypoint_offsets = {
            tid: getattr(task_fns, TASK_IDX_TO_NAME[tid]).object_keypoint_offsets() for tid in self.unique_task_ids
        }
        self.num_keypoints = max(len(v) for v in self.task_keypoint_offsets.values())

        self.allegro_fingertips = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        self.fingertip_offsets = np.array(
            [[0.05, 0.005, 0], [0.05, 0.005, 0], [0.05, 0.005, 0], [0.06, 0.005, 0]], dtype=np.float32
        )
        self.palm_offset = np.array([-0.00, -0.02, 0.16], dtype=np.float32)
        assert self.num_allegro_fingertips == len(self.allegro_fingertips)

        # can be only "full_state"
        self.obs_type = self.cfg["env"]["observationType"]
        if self.obs_type not in ["full_state"]:
            raise Exception("Unknown type of observations!")
        print("Obs type:", self.obs_type)

        num_dof_pos = self.num_hand_arm_dofs
        num_dof_vel = self.num_hand_arm_dofs
        num_dof_forces = self.num_hand_arm_dofs if self.with_dof_force_sensors else 0

        palm_pos_size = 3
        palm_rot_vel_angvel_size = 10
        obj_rot_vel_angvel_size = 10
        fingertip_rel_pos_size = 3 * self.num_allegro_fingertips
        keypoint_info_size = self.num_keypoints * 3 + self.num_keypoints * 3
        object_scales_size = 3
        max_keypoint_dist_size = 1
        lifted_object_flag_size = 1
        progress_obs_size = 1 + 1
        closest_fingertip_distance_size = self.num_allegro_fingertips
        reward_obs_size = 1

        self.full_state_size = (
            num_dof_pos
            + num_dof_vel
            + num_dof_forces
            + palm_pos_size
            + palm_rot_vel_angvel_size
            + obj_rot_vel_angvel_size
            + fingertip_rel_pos_size
            + keypoint_info_size
            + object_scales_size
            + max_keypoint_dist_size
            + lifted_object_flag_size
            + progress_obs_size
            + closest_fingertip_distance_size
            + reward_obs_size
        )

        num_obs = self.full_state_size
        self.task_embedding_enabled = self.cfg["env"].get("taskEmbedding", False)
        if self.task_embedding_enabled:
            num_obs += len(set(self.task_idx))

        num_states = self.full_state_size

        self.num_obs_dict = {"full_state": self.full_state_size}

        self.up_axis = "z"
        self.fingertip_obs = True

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = self.num_allegro_kuka_actions

        self.cfg["device_type"] = sim_device.split(":")[0] if sim_device.find(":") != -1 else sim_device
        self.cfg["device_id"] = int(sim_device.split(":")[1]) if sim_device.find(":") != -1 else 0
        self.cfg["headless"] = headless

        super().__init__(
            config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
            headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render,
        )

        if self.viewer is not None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # volume to sample target position from
        target_volume_origin = np.array([0, 0.05, 0.8], dtype=np.float32)
        target_volume_extent = np.array([[-0.4, 0.4], [-0.05, 0.3], [-0.12, 0.25]], dtype=np.float32)
        self.target_volume_origin = torch.from_numpy(target_volume_origin).to(self.device).float()
        self.target_volume_extent = torch.from_numpy(target_volume_extent).to(self.device).float()

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.hand_arm_default_dof_pos = torch.zeros(self.num_hand_arm_dofs, dtype=torch.float, device=self.device)
        desired_kuka_pos = torch.tensor([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358])  # pose v1
        self.hand_arm_default_dof_pos[:7] = desired_kuka_pos

        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_hand_arm_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.set_actor_root_state_object_indices: List[Tensor] = []

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_successes = torch.zeros_like(self.successes)

        # true objective value for the whole episode, plus saving values for the previous episode
        self.true_objective = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_true_objective = torch.zeros_like(self.true_objective)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(self.num_envs, device=self.device)
            + torch.log(self.force_prob_range[1])
        )

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.action_torques = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        self.obj_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3), dtype=torch.float, device=self.device
        )
        self.goal_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3), dtype=torch.float, device=self.device
        )

        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.closest_keypoint_max_dist = -torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_total_episode_closest_keypoint_max_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.total_episode_closest_keypoint_max_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_closest_keypoint_max_dist = 1000 * torch.ones(self.num_envs, dtype=torch.float, device=self.device)

        self.closest_fingertip_dist = -torch.ones(
            [self.num_envs, self.num_allegro_fingertips], dtype=torch.float, device=self.device
        )
        self.furthest_hand_dist = -torch.ones([self.num_envs], dtype=torch.float, device=self.device)

        self.finger_rew_coeffs = torch.ones(
            [self.num_envs, self.num_allegro_fingertips], dtype=torch.float, device=self.device
        )

        reward_keys = [
            "raw_fingertip_delta_rew",
            "raw_hand_delta_penalty",
            "raw_lifting_rew",
            "raw_keypoint_rew",
            "fingertip_delta_rew",
            "hand_delta_penalty",
            "lifting_rew",
            "lift_bonus_rew",
            "keypoint_rew",
            "bonus_rew",
            "kuka_actions_penalty",
            "allegro_actions_penalty",
        ]
        self.rewards_episode = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in reward_keys
        }

        self.eval_stats: bool = self.cfg["env"]["evalStats"]

        # ------------------- Multi-task bookkeeping (mirrors FrankaBaseEnvV2) -------------------#
        self.task_indices = torch.tensor(self.env_id_to_task_id, device=self.device)

        # build per-env views of the per-task parameters
        self.task_max_episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.task_success_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.task_force_scale = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.success_tolerance_per_task = {}
        self.initial_tolerance_per_task = {}
        self.target_tolerance_per_task = {}
        self.last_curriculum_update_per_task = {}
        for tid in self.unique_task_ids:
            p = self.task_params[tid]
            mask = self.task_indices == tid
            self.task_max_episode_length[mask] = int(p["episodeLength"])
            self.task_success_steps[mask] = float(p["successSteps"])
            self.task_force_scale[mask] = float(p["forceScale"])
            self.initial_tolerance_per_task[tid] = float(p["successTolerance"])
            self.target_tolerance_per_task[tid] = float(p["targetSuccessTolerance"])
            self.success_tolerance_per_task[tid] = float(p["successTolerance"])
            self.last_curriculum_update_per_task[tid] = 0
        self._refresh_success_tolerance_tensor()

        if self.task_embedding_enabled:
            transformed_indices = transform_task_indices(self.task_indices)
            num_unique_tasks = len(torch.unique(transformed_indices))
            self.task_embedding = torch.nn.functional.one_hot(transformed_indices, num_unique_tasks).float()

        self.extras["episode_cumulative"] = {}
        self.extras["task_indices"] = transform_task_indices(self.task_indices)
        ordered_task_names = [self.task_idx2name[tid.item()] for tid in torch.unique(self.task_indices)]
        self.extras["ordered_task_names"] = list(map(lambda s: s.replace("_", "-"), ordered_task_names))

        self.cumulatives = defaultdict(lambda: torch.zeros(self.num_envs, device=self.device))

        # initialize everything to a valid starting state
        self.reset_idx(torch.arange(self.num_envs, dtype=torch.long, device=self.device))
        self.set_actor_root_state_tensor_indexed()

    @property
    def task_idx2name(self):
        return TASK_IDX_TO_NAME

    def _resolve_task_params(self) -> dict:
        """Build {task_id: {param: value}} from env-level defaults overridden by env.taskParams[name]."""
        envc = self.cfg["env"]
        tp = envc.get("taskParams", None)
        defaults = {
            "episodeLength": envc["episodeLength"],
            "successSteps": envc["successSteps"],
            "forceScale": envc.get("forceScale", 0.0),
            "successTolerance": envc["successTolerance"],
            "targetSuccessTolerance": envc["targetSuccessTolerance"],
            "withSmallCuboids": envc["withSmallCuboids"],
            "withBigCuboids": envc["withBigCuboids"],
            "withSticks": envc["withSticks"],
        }
        params = {}
        for tid in self.unique_task_ids:
            name = TASK_IDX_TO_NAME[tid]
            p = dict(defaults)
            if tp is not None and name in tp:
                for k in PER_TASK_PARAM_KEYS:
                    if k in tp[name]:
                        p[k] = tp[name][k]
            params[tid] = p
        return params

    def _refresh_success_tolerance_tensor(self):
        st = torch.empty(self.num_envs, dtype=torch.float, device=self.device)
        for tid in self.unique_task_ids:
            st[self.task_indices == tid] = self.success_tolerance_per_task[tid]
        self.success_tolerance = st

    # ------------------------------------------------------------------ #
    #                       sim / env construction                       #
    # ------------------------------------------------------------------ #
    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _box_asset_files_and_scales(self, object_assets_root, generated_assets_dir, with_small, with_big, with_sticks):
        files = []
        scales = []
        try:
            for fname in os.listdir(generated_assets_dir):
                if fname.endswith(".urdf"):
                    os.remove(join(generated_assets_dir, fname))
        except Exception as exc:
            print(f"Exception {exc} while removing older procedurally-generated urdf assets")

        objects_rel_path = os.path.dirname(self.asset_files_dict[self.object_type])
        objects_dir = join(object_assets_root, objects_rel_path)
        base_mesh = join(objects_dir, "meshes", "cube_multicolor.obj")

        generate_default_cube(generated_assets_dir, base_mesh, self.object_base_size)
        if with_small:
            generate_small_cuboids(generated_assets_dir, base_mesh, self.object_base_size)
        if with_big:
            generate_big_cuboids(generated_assets_dir, base_mesh, self.object_base_size)
        if with_sticks:
            generate_sticks(generated_assets_dir, base_mesh, self.object_base_size)

        for fname in sorted(os.listdir(generated_assets_dir)):
            if fname.endswith(".urdf"):
                scale_tokens = os.path.splitext(fname)[0].split("_")[2:]
                files.append(join(generated_assets_dir, fname))
                scales.append([float(scale_token) / 100 for scale_token in scale_tokens])
        return files, scales

    def _generate_task_object_assets(self, object_asset_root):
        """Procedurally generate a (possibly task-specific) object set per task."""
        self.task_object_asset_files = {}
        self.task_object_asset_scales = {}
        self._tmp_asset_dirs = []
        for tid in self.unique_task_ids:
            p = self.task_params[tid]
            tmp = tempfile.TemporaryDirectory()
            self._tmp_asset_dirs.append(tmp)
            files, scales = self._box_asset_files_and_scales(
                object_asset_root, tmp.name, p["withSmallCuboids"], p["withBigCuboids"], p["withSticks"]
            )
            if not self.randomize_object_dimensions:
                files = files[:1]
                scales = scales[:1]
            # fixed seed so the object-type distribution is reproducible across restarts
            files_and_scales = list(zip(files, scales))
            rng = np.random.default_rng(42)
            rng.shuffle(files_and_scales)
            files, scales = zip(*files_and_scales)
            self.task_object_asset_files[tid] = list(files)
            self.task_object_asset_scales[tid] = list(scales)

    def _load_task_object_assets(self):
        self.task_object_assets = {}
        object_rb_count = object_shapes_count = None
        for tid in self.unique_task_ids:
            opts = gymapi.AssetOptions()
            assets = []
            for f in self.task_object_asset_files[tid]:
                assets.append(self.gym.load_asset(self.sim, os.path.dirname(f), os.path.basename(f), opts))
            self.task_object_assets[tid] = assets
            if object_rb_count is None:
                object_rb_count = self.gym.get_asset_rigid_body_count(assets[0])
                object_shapes_count = self.gym.get_asset_rigid_shape_count(assets[0])
        return object_rb_count, object_shapes_count

    def _object_start_pose(self, allegro_pose, table_pose_dy, table_pose_dz):
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = allegro_pose.p.x
        pose_dy, pose_dz = table_pose_dy, table_pose_dz + 0.25
        object_start_pose.p.y = allegro_pose.p.y + pose_dy
        object_start_pose.p.z = allegro_pose.p.z + pose_dz
        return object_start_pose

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../assets")
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root)
            )
        object_asset_root = asset_root

        # procedurally generate + load the (per-task) object sets
        self._generate_task_object_assets(object_asset_root)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        print(f"Loading asset {self.hand_arm_asset_file} from {asset_root}")
        allegro_kuka_asset = self.gym.load_asset(self.sim, asset_root, self.hand_arm_asset_file, asset_options)

        self.num_hand_arm_bodies = self.gym.get_asset_rigid_body_count(allegro_kuka_asset)
        self.num_hand_arm_shapes = self.gym.get_asset_rigid_shape_count(allegro_kuka_asset)
        num_hand_arm_dofs = self.gym.get_asset_dof_count(allegro_kuka_asset)
        assert self.num_hand_arm_dofs == num_hand_arm_dofs, (
            f"Number of DOFs in asset {allegro_kuka_asset} is {num_hand_arm_dofs}, expected {self.num_hand_arm_dofs}"
        )

        max_agg_bodies = self.num_hand_arm_bodies
        max_agg_shapes = self.num_hand_arm_shapes

        allegro_hand_dof_props = self.gym.get_asset_dof_properties(allegro_kuka_asset)
        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        for i in range(self.num_hand_arm_dofs):
            self.arm_hand_dof_lower_limits.append(allegro_hand_dof_props["lower"][i])
            self.arm_hand_dof_upper_limits.append(allegro_hand_dof_props["upper"][i])
        self.arm_hand_dof_lower_limits = to_torch(self.arm_hand_dof_lower_limits, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(self.arm_hand_dof_upper_limits, device=self.device)

        allegro_pose = gymapi.Transform()
        allegro_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx)) + gymapi.Vec3(0.0, 0.8, 0)
        allegro_pose.r = gymapi.Quat(0, 0, 0, 1)

        object_rb_count, object_shapes_count = self._load_task_object_assets()
        max_agg_bodies += object_rb_count
        max_agg_shapes += object_shapes_count

        # table
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = False
        table_asset_options.fix_base_link = True
        table_asset = self.gym.load_asset(self.sim, asset_root, self.asset_files_dict["table"], table_asset_options)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3()
        table_pose.p.x = allegro_pose.p.x
        table_pose_dy, table_pose_dz = -0.8, 0.38
        table_pose.p.y = allegro_pose.p.y + table_pose_dy
        table_pose.p.z = allegro_pose.p.z + table_pose_dz

        table_rb_count = self.gym.get_asset_rigid_body_count(table_asset)
        table_shapes_count = self.gym.get_asset_rigid_shape_count(table_asset)
        max_agg_bodies += table_rb_count
        max_agg_shapes += table_shapes_count

        # ---- load per-task additional (goal) assets once; size aggregates to the max ----
        self.object_start_pose = self._object_start_pose(allegro_pose, table_pose_dy, table_pose_dz)
        max_additional_rb = 0
        max_additional_shapes = 0
        for tid in self.unique_task_ids:
            name = self.task_idx2name[tid]
            # reorientation loads goal cubes that must match its own object pool
            self._loading_task_files = self.task_object_asset_files[tid]
            additional_rb, additional_shapes = getattr(task_fns, name).load_additional_assets(
                self, object_asset_root, allegro_pose
            )
            max_additional_rb = max(max_additional_rb, additional_rb)
            max_additional_shapes = max(max_additional_shapes, additional_shapes)
        max_agg_bodies += max_additional_rb
        max_agg_shapes += max_additional_shapes

        self.allegro_hands = []
        self.envs = []
        object_init_state = []
        self.rigid_body_name_to_idx = {}
        self.allegro_hand_indices = []
        object_indices = []
        table_indices = []
        object_scales = []
        object_keypoint_offsets = []
        # one extra "goal" actor per env (ball / bucket / goal-cube), filled by the task hooks
        self.goal_object_indices = []

        self.allegro_fingertip_handles = [
            self.gym.find_asset_rigid_body_index(allegro_kuka_asset, name) for name in self.allegro_fingertips
        ]
        self.allegro_palm_handle = self.gym.find_asset_rigid_body_index(allegro_kuka_asset, "iiwa7_link_7")
        # objects are added right after the arm in terms of create_actor()
        self.object_rb_handles = list(range(self.num_hand_arm_bodies, self.num_hand_arm_bodies + object_rb_count))

        # per-task running counter to index into each task's object pool
        task_local_counter = {tid: 0 for tid in self.unique_task_ids}

        for i in range(self.num_envs):
            tid = self.env_id_to_task_id[i]
            task_name = self.task_idx2name[tid]

            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            allegro_actor = self.gym.create_actor(env_ptr, allegro_kuka_asset, allegro_pose, "allegro", i, -1, 0)
            populate_dof_properties(allegro_hand_dof_props, self.dof_params, self.num_arm_dofs, self.num_hand_dofs)
            self.gym.set_actor_dof_properties(env_ptr, allegro_actor, allegro_hand_dof_props)
            allegro_hand_idx = self.gym.get_actor_index(env_ptr, allegro_actor, gymapi.DOMAIN_SIM)
            self.allegro_hand_indices.append(allegro_hand_idx)
            if i == 0:
                for name in self.gym.get_actor_rigid_body_names(env_ptr, allegro_actor):
                    self.rigid_body_name_to_idx["allegro/" + name] = self.gym.find_actor_rigid_body_index(
                        env_ptr, allegro_actor, name, gymapi.DOMAIN_ENV
                    )

            # add the manipulated object (cube) from this task's pool
            local_idx = task_local_counter[tid]
            task_local_counter[tid] += 1
            pool = self.task_object_assets[tid]
            object_asset_idx = local_idx % len(pool)
            object_asset = pool[object_asset_idx]
            object_handle = self.gym.create_actor(env_ptr, object_asset, self.object_start_pose, "object", i, 0, 0)
            object_init_state.append(
                [
                    self.object_start_pose.p.x, self.object_start_pose.p.y, self.object_start_pose.p.z,
                    self.object_start_pose.r.x, self.object_start_pose.r.y, self.object_start_pose.r.z,
                    self.object_start_pose.r.w, 0, 0, 0, 0, 0, 0,
                ]
            )
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            object_indices.append(object_idx)
            if i == 0:
                for name in self.gym.get_actor_rigid_body_names(env_ptr, object_handle):
                    self.rigid_body_name_to_idx["object/" + name] = self.gym.find_actor_rigid_body_index(
                        env_ptr, object_handle, name, gymapi.DOMAIN_ENV
                    )

            object_scale = self.task_object_asset_scales[tid][object_asset_idx]
            object_scales.append(object_scale)
            # build per-env keypoint offsets, padded (by replication) to self.num_keypoints
            task_offsets = self.task_keypoint_offsets[tid]
            padded_offsets = [copy(task_offsets[k % len(task_offsets)]) for k in range(self.num_keypoints)]
            env_offsets = []
            for keypoint in padded_offsets:
                keypoint = copy(keypoint)
                for coord_idx in range(3):
                    keypoint[coord_idx] *= object_scale[coord_idx] * self.object_base_size * self.keypoint_scale / 2
                env_offsets.append(keypoint)
            object_keypoint_offsets.append(env_offsets)

            # table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table_object", i, 0, 0)
            table_object_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            table_indices.append(table_object_idx)
            if i == 0:
                for name in self.gym.get_actor_rigid_body_names(env_ptr, table_handle):
                    self.rigid_body_name_to_idx["table/" + name] = self.gym.find_actor_rigid_body_index(
                        env_ptr, table_handle, name, gymapi.DOMAIN_ENV
                    )

            # task-specific extra object (goal cube / ball / bucket)
            getattr(task_fns, task_name).create_additional_objects(
                self, env_ptr, env_idx=i, object_asset_idx=object_asset_idx
            )

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_actor)

        object_rb_props = self.gym.get_actor_rigid_body_properties(self.envs[0], object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13
        )
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()

        self.allegro_fingertip_handles = to_torch(self.allegro_fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.allegro_hand_indices = to_torch(self.allegro_hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(table_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

        self.object_scales = to_torch(object_scales, dtype=torch.float, device=self.device)
        self.object_keypoint_offsets = to_torch(object_keypoint_offsets, dtype=torch.float, device=self.device)

        for tmp in getattr(self, "_tmp_asset_dirs", []):
            try:
                tmp.cleanup()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #                              rewards                                #
    # ------------------------------------------------------------------ #
    def _distance_delta_rewards(self, lifted_object: Tensor) -> Tuple[Tensor, Tensor]:
        fingertip_deltas_closest = self.closest_fingertip_dist - self.curr_fingertip_distances
        self.closest_fingertip_dist = torch.minimum(self.closest_fingertip_dist, self.curr_fingertip_distances)

        hand_deltas_furthest = self.furthest_hand_dist - self.curr_fingertip_distances[:, 0]
        self.furthest_hand_dist = torch.maximum(self.furthest_hand_dist, self.curr_fingertip_distances[:, 0])

        fingertip_deltas = torch.clip(fingertip_deltas_closest, 0, 10)
        fingertip_deltas *= self.finger_rew_coeffs
        fingertip_delta_rew = torch.sum(fingertip_deltas, dim=-1)
        fingertip_delta_rew *= ~lifted_object

        hand_delta_penalty = torch.clip(hand_deltas_furthest, -10, 0)
        hand_delta_penalty *= ~lifted_object
        hand_delta_penalty *= self.num_allegro_fingertips
        return fingertip_delta_rew, hand_delta_penalty

    def _lifting_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        z_lift = 0.05 + self.object_pos[:, 2] - self.object_init_state[:, 2]
        lifting_rew = torch.clip(z_lift, 0, 0.5)
        lifted_object = (z_lift > self.lifting_bonus_threshold) | self.lifted_object
        just_lifted_above_threshold = lifted_object & ~self.lifted_object
        lift_bonus_rew = self.lifting_bonus * just_lifted_above_threshold
        lifting_rew *= ~lifted_object
        self.lifted_object = lifted_object
        return lifting_rew, lift_bonus_rew, lifted_object

    def _keypoint_reward(self, lifted_object: Tensor) -> Tensor:
        max_keypoint_deltas = self.closest_keypoint_max_dist - self.keypoints_max_dist
        self.closest_keypoint_max_dist = torch.minimum(self.closest_keypoint_max_dist, self.keypoints_max_dist)
        max_keypoint_deltas = torch.clip(max_keypoint_deltas, 0, 100)
        keypoint_rew = max_keypoint_deltas * lifted_object
        return keypoint_rew

    def _action_penalties(self) -> Tuple[Tensor, Tensor]:
        kuka_actions_penalty = (
            torch.sum(torch.abs(self.arm_hand_dof_vel[..., 0:7]), dim=-1) * self.kuka_actions_penalty_scale
        )
        allegro_actions_penalty = (
            torch.sum(torch.abs(self.arm_hand_dof_vel[..., 7 : self.num_hand_arm_dofs]), dim=-1)
            * self.allegro_actions_penalty_scale
        )
        return -1 * kuka_actions_penalty, -1 * allegro_actions_penalty

    def _compute_resets(self, is_success):
        resets = torch.where(self.object_pos[:, 2] < 0.1, torch.ones_like(self.reset_buf), self.reset_buf)  # fall
        if self.max_consecutive_successes > 0:
            self.progress_buf = torch.where(is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf)
            resets = torch.where(self.successes >= self.max_consecutive_successes, torch.ones_like(resets), resets)
        # per-task episode length
        resets = torch.where(
            self.progress_buf >= self.task_max_episode_length - 1, torch.ones_like(resets), resets
        )
        resets = self._extra_reset_rules(resets)
        return resets

    def _true_objective(self) -> Tensor:
        # identical functional form across tasks, but evaluated with each task's own tolerances
        true_obj = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for tid in self.unique_task_ids:
            mask = self.task_indices == tid
            true_obj[mask] = tolerance_successes_objective(
                self.success_tolerance_per_task[tid],
                self.initial_tolerance_per_task[tid],
                self.target_tolerance_per_task[tid],
                self.successes[mask],
            )
        return true_obj

    def _extra_reset_rules(self, resets):
        # dispatch to the per-task rule, masked to that task's env block
        for tid in self.unique_task_ids:
            name = self.task_idx2name[tid]
            rule = getattr(getattr(task_fns, name), "extra_reset_rules", None)
            if rule is None:
                continue
            mask = self.task_indices == tid
            resets = rule(self, resets, mask)
        return resets

    def _extra_curriculum(self):
        # success_tolerance is shared per task; advance each task's curriculum independently
        updated = False
        for tid in self.unique_task_ids:
            mask = self.task_indices == tid
            new_tol, new_upd = tolerance_curriculum(
                self.last_curriculum_update_per_task[tid],
                self.frame_since_restart,
                self.tolerance_curriculum_interval,
                self.prev_episode_successes[mask],
                self.success_tolerance_per_task[tid],
                self.initial_tolerance_per_task[tid],
                self.target_tolerance_per_task[tid],
                self.tolerance_curriculum_increment,
            )
            if new_tol != self.success_tolerance_per_task[tid] or new_upd != self.last_curriculum_update_per_task[tid]:
                updated = True
            self.success_tolerance_per_task[tid] = new_tol
            self.last_curriculum_update_per_task[tid] = new_upd
        if updated:
            self._refresh_success_tolerance_tensor()

    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew, hand_delta_penalty = self._distance_delta_rewards(lifted_object)
        keypoint_rew = self._keypoint_reward(lifted_object)

        # per-env success tolerance (per task, advanced by curriculum)
        keypoint_success_tolerance = self.success_tolerance * self.keypoint_scale
        near_goal: Tensor = self.keypoints_max_dist <= keypoint_success_tolerance
        self.near_goal_steps += near_goal

        # per-task number of consecutive near-goal steps required for a success
        is_success = self.near_goal_steps >= self.task_success_steps
        goal_resets = is_success
        self.successes += is_success
        self.reset_goal_buf[:] = goal_resets

        self.rewards_episode["raw_fingertip_delta_rew"] += fingertip_delta_rew
        self.rewards_episode["raw_hand_delta_penalty"] += hand_delta_penalty
        self.rewards_episode["raw_lifting_rew"] += lifting_rew
        self.rewards_episode["raw_keypoint_rew"] += keypoint_rew

        fingertip_delta_rew *= self.distance_delta_rew_scale
        hand_delta_penalty *= self.distance_delta_rew_scale * 0  # currently disabled
        lifting_rew *= self.lifting_rew_scale
        keypoint_rew *= self.keypoint_rew_scale

        kuka_actions_penalty, allegro_actions_penalty = self._action_penalties()
        # spread the goal-reaching bonus over each task's required success steps
        bonus_rew = near_goal * (self.reach_goal_bonus / self.task_success_steps)

        reward = (
            fingertip_delta_rew
            + hand_delta_penalty
            + lifting_rew
            + lift_bonus_rew
            + keypoint_rew
            + kuka_actions_penalty
            + allegro_actions_penalty
            + bonus_rew
        )
        self.rew_buf[:] = reward

        resets = self._compute_resets(is_success)
        self.reset_buf[:] = resets

        self.extras["successes"] = self.prev_episode_successes
        self.extras["closest_keypoint_max_dist"] = self.prev_episode_closest_keypoint_max_dist
        self.true_objective = self._true_objective()
        self.extras["true_objective"] = self.true_objective

        # Per-task successes: the success criterion (goal-hold for `successSteps`)
        # differs per task, so the aggregate `successes` mixes incomparable counts.
        # These per-task means are 0-dim tensors -> logged as scalars by the observer.
        for tid in self.unique_task_ids:
            name = TASK_IDX_TO_NAME[tid]
            mask = self.task_indices == tid
            self.extras[f"successes/{name}"] = self.prev_episode_successes[mask].mean()

        rewards = [
            (fingertip_delta_rew, "fingertip_delta_rew"),
            (hand_delta_penalty, "hand_delta_penalty"),
            (lifting_rew, "lifting_rew"),
            (lift_bonus_rew, "lift_bonus_rew"),
            (keypoint_rew, "keypoint_rew"),
            (kuka_actions_penalty, "kuka_actions_penalty"),
            (allegro_actions_penalty, "allegro_actions_penalty"),
            (bonus_rew, "bonus_rew"),
        ]
        episode_cumulative = dict()
        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
            episode_cumulative[rew_name] = rew_value
        self.extras["rewards_episode"] = self.rewards_episode
        self.extras["episode_cumulative"] = episode_cumulative

        return self.rew_buf, is_success

    # ------------------------------------------------------------------ #
    #                            observations                            #
    # ------------------------------------------------------------------ #
    def compute_observations(self) -> Tuple[Tensor, int]:
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.object_state = self.root_state_tensor[self.object_indices, 0:13]
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.palm_center_offset = torch.from_numpy(self.palm_offset).to(self.device).repeat((self.num_envs, 1))
        self._palm_state = self.rigid_body_states[:, self.allegro_palm_handle][:, 0:13]
        self._palm_pos = self.rigid_body_states[:, self.allegro_palm_handle][:, 0:3]
        self._palm_rot = self.rigid_body_states[:, self.allegro_palm_handle][:, 3:7]
        self.palm_center_pos = self._palm_pos + quat_rotate(self._palm_rot, self.palm_center_offset)

        self.fingertip_state = self.rigid_body_states[:, self.allegro_fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.allegro_fingertip_handles][:, :, 0:3]
        self.fingertip_rot = self.rigid_body_states[:, self.allegro_fingertip_handles][:, :, 3:7]

        if not isinstance(self.fingertip_offsets, torch.Tensor):
            self.fingertip_offsets = (
                torch.from_numpy(self.fingertip_offsets).to(self.device).repeat((self.num_envs, 1, 1))
            )

        if hasattr(self, "fingertip_pos_rel_object"):
            self.fingertip_pos_rel_object_prev[:, :, :] = self.fingertip_pos_rel_object
        else:
            self.fingertip_pos_rel_object_prev = None

        self.fingertip_pos_offset = torch.zeros_like(self.fingertip_pos).to(self.device)
        for i in range(self.num_allegro_fingertips):
            self.fingertip_pos_offset[:, i] = self.fingertip_pos[:, i] + quat_rotate(
                self.fingertip_rot[:, i], self.fingertip_offsets[:, i]
            )

        obj_pos_repeat = self.object_pos.unsqueeze(1).repeat(1, self.num_allegro_fingertips, 1)
        self.fingertip_pos_rel_object = self.fingertip_pos_offset - obj_pos_repeat
        self.curr_fingertip_distances = torch.norm(self.fingertip_pos_rel_object, dim=-1)

        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0, self.curr_fingertip_distances, self.closest_fingertip_dist
        )
        self.furthest_hand_dist = torch.where(
            self.furthest_hand_dist < 0.0, self.curr_fingertip_distances[:, 0], self.furthest_hand_dist
        )

        palm_center_repeat = self.palm_center_pos.unsqueeze(1).repeat(1, self.num_allegro_fingertips, 1)
        self.fingertip_pos_rel_palm = self.fingertip_pos_offset - palm_center_repeat

        if self.fingertip_pos_rel_object_prev is None:
            self.fingertip_pos_rel_object_prev = self.fingertip_pos_rel_object.clone()

        for i in range(self.num_keypoints):
            self.obj_keypoint_pos[:, i] = self.object_pos + quat_rotate(
                self.object_rot, self.object_keypoint_offsets[:, i]
            )
            self.goal_keypoint_pos[:, i] = self.goal_pos + quat_rotate(
                self.goal_rot, self.object_keypoint_offsets[:, i]
            )

        self.keypoints_rel_goal = self.obj_keypoint_pos - self.goal_keypoint_pos

        palm_center_repeat = self.palm_center_pos.unsqueeze(1).repeat(1, self.num_keypoints, 1)
        self.keypoints_rel_palm = self.obj_keypoint_pos - palm_center_repeat

        self.keypoint_distances_l2 = torch.norm(self.keypoints_rel_goal, dim=-1)
        self.keypoints_max_dist = self.keypoint_distances_l2.max(dim=-1).values

        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0, self.keypoints_max_dist, self.closest_keypoint_max_dist
        )

        full_state_size, reward_obs_ofs = self.compute_full_state(self.obs_buf)
        assert full_state_size == self.full_state_size, (
            f"Expected full state size {self.full_state_size}, actual: {full_state_size}"
        )

        if self.task_embedding_enabled:
            self.obs_buf[:, self.full_state_size :] = self.task_embedding

        return self.obs_buf, reward_obs_ofs

    def compute_full_state(self, buf: Tensor) -> Tuple[int, int]:
        num_dofs = self.num_hand_arm_dofs
        ofs = 0

        buf[:, ofs : ofs + num_dofs] = unscale(
            self.arm_hand_dof_pos[:, :num_dofs],
            self.arm_hand_dof_lower_limits[:num_dofs],
            self.arm_hand_dof_upper_limits[:num_dofs],
        )
        ofs += num_dofs

        buf[:, ofs : ofs + num_dofs] = self.arm_hand_dof_vel[:, :num_dofs]
        ofs += num_dofs

        buf[:, ofs : ofs + 3] = self.palm_center_pos
        ofs += 3

        buf[:, ofs : ofs + 10] = self._palm_state[:, 3:13]
        ofs += 10

        buf[:, ofs : ofs + 10] = self.object_state[:, 3:13]
        ofs += 10

        fingertip_rel_pos_size = 3 * self.num_allegro_fingertips
        buf[:, ofs : ofs + fingertip_rel_pos_size] = self.fingertip_pos_rel_palm.reshape(
            self.num_envs, fingertip_rel_pos_size
        )
        ofs += fingertip_rel_pos_size

        keypoint_rel_pos_size = 3 * self.num_keypoints
        buf[:, ofs : ofs + keypoint_rel_pos_size] = self.keypoints_rel_palm.reshape(
            self.num_envs, keypoint_rel_pos_size
        )
        ofs += keypoint_rel_pos_size

        buf[:, ofs : ofs + keypoint_rel_pos_size] = self.keypoints_rel_goal.reshape(
            self.num_envs, keypoint_rel_pos_size
        )
        ofs += keypoint_rel_pos_size

        buf[:, ofs : ofs + 3] = self.object_scales
        ofs += 3

        buf[:, ofs : ofs + 1] = self.closest_keypoint_max_dist.unsqueeze(-1)
        ofs += 1

        buf[:, ofs : ofs + self.num_allegro_fingertips] = self.closest_fingertip_dist
        ofs += self.num_allegro_fingertips

        buf[:, ofs : ofs + 1] = self.lifted_object.unsqueeze(-1)
        ofs += 1

        buf[:, ofs : ofs + 1] = torch.log(self.progress_buf / 10 + 1).unsqueeze(-1)
        ofs += 1
        buf[:, ofs : ofs + 1] = torch.log(self.successes + 1).unsqueeze(-1)
        ofs += 1

        reward_obs_ofs = ofs
        ofs += 1

        assert ofs == self.full_state_size
        return ofs, reward_obs_ofs

    def clamp_obs(self, obs_buf: Tensor) -> None:
        if self.clamp_abs_observations > 0:
            obs_buf[:, : self.full_state_size].clamp_(-self.clamp_abs_observations, self.clamp_abs_observations)

    # ------------------------------------------------------------------ #
    #                               resets                               #
    # ------------------------------------------------------------------ #
    def get_random_quat(self, env_ids):
        uvw = torch_rand_float(0, 1.0, (len(env_ids), 3), device=self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        q_y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
        q_z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
        new_rot = torch.cat((q_x.unsqueeze(-1), q_y.unsqueeze(-1), q_z.unsqueeze(-1), q_w.unsqueeze(-1)), dim=-1)
        return new_rot

    def _reset_target(self, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True) -> None:
        if len(env_ids) == 0:
            return
        # group env_ids by task and dispatch to the per-task target reset
        task_of_env = self.task_indices[env_ids]
        for tid in self.unique_task_ids:
            name = self.task_idx2name[tid]
            ids = env_ids[task_of_env == tid]
            if len(ids) == 0:
                continue
            getattr(task_fns, name).reset_target(self, ids, reset_buf_idxs, tensor_reset)

    def reset_target_pose(self, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True) -> None:
        self._reset_target(env_ids, reset_buf_idxs, tensor_reset=tensor_reset)
        if tensor_reset:
            self.reset_goal_buf[env_ids] = 0
            self.near_goal_steps[env_ids] = 0
            self.prev_total_episode_closest_keypoint_max_dist[env_ids] = self.total_episode_closest_keypoint_max_dist[env_ids]
            self.total_episode_closest_keypoint_max_dist[env_ids] += torch.where(
                self.closest_keypoint_max_dist[env_ids] > 0,
                self.closest_keypoint_max_dist[env_ids],
                torch.zeros_like(self.closest_keypoint_max_dist[env_ids]),
            )
            self.closest_keypoint_max_dist[env_ids] = -1

    def reset_object_pose(self, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True):
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            obj_indices = self.object_indices[env_ids]
            rand_pos_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device)
            self.root_state_tensor[obj_indices] = self.object_init_state[env_ids].clone()
            self.root_state_tensor[obj_indices, 0:1] = (
                self.object_init_state[env_ids, 0:1] + self.reset_position_noise_x * rand_pos_floats[:, 0:1]
            )
            self.root_state_tensor[obj_indices, 1:2] = (
                self.object_init_state[env_ids, 1:2] + self.reset_position_noise_y * rand_pos_floats[:, 1:2]
            )
            self.root_state_tensor[obj_indices, 2:3] = (
                self.object_init_state[env_ids, 2:3] + self.reset_position_noise_z * rand_pos_floats[:, 2:3]
            )
            new_object_rot = self.get_random_quat(env_ids)
            self.root_state_tensor[obj_indices, 3:7] = new_object_rot
            self.root_state_tensor[obj_indices, 7:13] = torch.zeros_like(self.root_state_tensor[obj_indices, 7:13])

        if tensor_reset:
            self.closest_fingertip_dist[env_ids] = -1
            self.furthest_hand_dist[env_ids] = -1
            self.lifted_object[env_ids] = False
        self.deferred_set_actor_root_state_tensor_indexed([self.object_indices[env_ids]])

    def deferred_set_actor_root_state_tensor_indexed(self, obj_indices: List[Tensor]) -> None:
        self.set_actor_root_state_object_indices.extend(obj_indices)

    def set_actor_root_state_tensor_indexed(self) -> None:
        object_indices: List[Tensor] = self.set_actor_root_state_object_indices
        if not object_indices:
            return
        unique_object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(unique_object_indices),
            len(unique_object_indices),
        )
        self.set_actor_root_state_object_indices = []

    def _extra_object_indices(self, env_ids: Tensor) -> List[Tensor]:
        # one extra goal/bucket actor per env, regardless of task
        return [self.goal_object_indices[env_ids]]

    def reset_idx(self, env_ids: Tensor, reset_buf_idxs=None, episode_reset=True, tensor_reset=True) -> None:
        if len(env_ids) == 0:
            return

        if self.randomize and episode_reset:
            self.apply_randomizations(self.randomization_params)

        self.reset_target_pose(env_ids, reset_buf_idxs, tensor_reset=tensor_reset)

        if tensor_reset:
            self.rb_forces[env_ids, :, :] = 0.0

        self.reset_object_pose(env_ids, reset_buf_idxs, tensor_reset=tensor_reset)

        hand_indices = self.allegro_hand_indices[env_ids].to(torch.int32)

        if tensor_reset:
            self.random_force_prob[env_ids] = torch.exp(
                (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                * torch.rand(len(env_ids), device=self.device)
                + torch.log(self.force_prob_range[1])
            )

        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            delta_max = self.arm_hand_dof_upper_limits - self.hand_arm_default_dof_pos
            delta_min = self.arm_hand_dof_lower_limits - self.hand_arm_default_dof_pos
            rand_dof_floats = torch_rand_float(0.0, 1.0, (len(env_ids), self.num_hand_arm_dofs), device=self.device)
            rand_delta = delta_min + (delta_max - delta_min) * rand_dof_floats

            noise_coeff = torch.zeros_like(self.hand_arm_default_dof_pos, device=self.device)
            noise_coeff[0:7] = self.reset_dof_pos_noise_arm
            noise_coeff[7 : self.num_hand_arm_dofs] = self.reset_dof_pos_noise_fingers

            allegro_pos = self.hand_arm_default_dof_pos + noise_coeff * rand_delta
            self.arm_hand_dof_pos[env_ids, :] = allegro_pos

            rand_vel_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_hand_arm_dofs), device=self.device)
            self.arm_hand_dof_vel[env_ids, :] = self.reset_dof_vel_noise * rand_vel_floats
            self.prev_targets[env_ids, : self.num_hand_arm_dofs] = allegro_pos
            self.cur_targets[env_ids, : self.num_hand_arm_dofs] = allegro_pos

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids)
        )

        self.deferred_set_actor_root_state_tensor_indexed(self._extra_object_indices(env_ids))

        if episode_reset and tensor_reset:
            self.progress_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0

            self.prev_episode_successes[env_ids] = self.successes[env_ids]
            self.successes[env_ids] = 0

            self.prev_episode_true_objective[env_ids] = self.true_objective[env_ids]
            self.true_objective[env_ids] = 0

            self.prev_episode_closest_keypoint_max_dist[env_ids] = torch.where(
                self.prev_episode_successes[env_ids] > 0,
                self.prev_total_episode_closest_keypoint_max_dist[env_ids] / self.prev_episode_successes[env_ids],
                self.total_episode_closest_keypoint_max_dist[env_ids],
            )
            self.total_episode_closest_keypoint_max_dist[env_ids] = 0
            self.prev_total_episode_closest_keypoint_max_dist[env_ids] = 0

            for key in self.rewards_episode.keys():
                self.rewards_episode[key][env_ids] = 0

            self.extras["scalars"] = dict()
            # break tolerance down per task -- each task advances its curriculum independently
            for tid in self.unique_task_ids:
                self.extras["scalars"][f"success_tolerance/{TASK_IDX_TO_NAME[tid]}"] = float(
                    self.success_tolerance_per_task[tid]
                )

    # ------------------------------------------------------------------ #
    #                            physics loop                            #
    # ------------------------------------------------------------------ #
    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        if self.privileged_actions:
            torque_actions = actions[:, :3]
            actions = actions[:, 3:]

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        reset_goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # goal-only resets: reset the target pose but not the whole episode
        combined = torch.cat([reset_env_ids, reset_goal_env_ids, reset_goal_env_ids])
        uniques, counts = combined.unique(return_counts=True)
        goal_only_env_ids = uniques[counts == 2]
        self.reset_target_pose(goal_only_env_ids, None)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids, None)

        self.set_actor_root_state_tensor_indexed()

        if self.use_relative_control:
            raise NotImplementedError("Use relative control False for now")
        else:
            self.cur_targets[:, 7 : self.num_hand_arm_dofs] = scale(
                actions[:, 7 : self.num_hand_arm_dofs],
                self.arm_hand_dof_lower_limits[7 : self.num_hand_arm_dofs],
                self.arm_hand_dof_upper_limits[7 : self.num_hand_arm_dofs],
            )
            self.cur_targets[:, 7 : self.num_hand_arm_dofs] = (
                self.act_moving_average * self.cur_targets[:, 7 : self.num_hand_arm_dofs]
                + (1.0 - self.act_moving_average) * self.prev_targets[:, 7 : self.num_hand_arm_dofs]
            )
            self.cur_targets[:, 7 : self.num_hand_arm_dofs] = tensor_clamp(
                self.cur_targets[:, 7 : self.num_hand_arm_dofs],
                self.arm_hand_dof_lower_limits[7 : self.num_hand_arm_dofs],
                self.arm_hand_dof_upper_limits[7 : self.num_hand_arm_dofs],
            )

            targets = self.prev_targets[:, :7] + self.hand_dof_speed_scale * self.dt * self.actions[:, :7]
            self.cur_targets[:, :7] = tensor_clamp(
                targets, self.arm_hand_dof_lower_limits[:7], self.arm_hand_dof_upper_limits[:7]
            )

        self.prev_targets[:, :] = self.cur_targets[:, :]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # random forces on the object (per-task force scale; throw disables them)
        if self.force_scale_max > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = (
                torch.randn(self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device)
                * self.object_rb_masses
                * self.task_force_scale[force_indices].unsqueeze(-1)
            )
            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE
            )

        if self.privileged_actions:
            torque_actions = torque_actions.unsqueeze(1)
            torque_actions *= self.privileged_actions_torque
            self.action_torques[:, self.object_rb_handles, :] = torque_actions
            self.gym.apply_rigid_body_force_tensors(
                self.sim, None, gymtorch.unwrap_tensor(self.action_torques), gymapi.ENV_SPACE
            )

    def post_physics_step(self):
        self.frame_since_restart += 1
        self.progress_buf += 1
        self.randomize_buf += 1

        self._extra_curriculum()

        obs_buf, reward_obs_ofs = self.compute_observations()
        rewards, is_success = self.compute_kuka_reward()

        reward_obs_scale = 0.01
        obs_buf[:, reward_obs_ofs : reward_obs_ofs + 1] = rewards.unsqueeze(-1) * reward_obs_scale
        self.clamp_obs(obs_buf)

        self._log_task_metrics(is_success)

    def step(self, actions):
        obs_dict, rew, reset, extras = super().step(actions)
        # VecTask computes time_outs from the scalar max_episode_length; recompute per task
        self.timeout_buf = (self.progress_buf >= self.task_max_episode_length - 1) & (self.reset_buf != 0)
        extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        return obs_dict, rew, reset, extras

    def _log_task_metrics(self, is_success):
        """Per-task episode reward / length into self.extras['episode'].

        We intentionally do NOT log a Meta-World-style binary success here. For
        AllegroKuka the meaningful metric is the *number* of consecutive goals
        reached, not whether >=1 goal was reached; SAPG's metrics (consecutive
        ``successes``, ``true_objective``, ``closest_keypoint_max_dist``) are
        exported globally in ``compute_kuka_reward`` and logged by the observer.
        """
        self.cumulatives["reward"] += self.rew_buf

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) == 0:
            return

        self.extras["episode"] = {}
        for tid in self.unique_task_ids:
            counted = torch.logical_and(self.task_indices == tid, self.reset_buf).nonzero(as_tuple=False).squeeze(-1)
            if len(counted) == 0:
                continue
            self.extras["episode"][f"task_{tid}_reward"] = self.cumulatives["reward"][counted].clone()
            self.extras["episode"][f"task_{tid}_eplength"] = self.progress_buf[counted].clone()

        self.cumulatives["reward"][env_ids] = 0
