# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import os
from typing import Tuple, Dict, Any
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor

from scipy.spatial.transform import Rotation as R
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.utils.attr_dict import AttrDict
from isaacgymenvs.tasks.utils.math import *
from isaacgymenvs.tasks.locomotion.terrain_utils import Terrain

def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def transform_task_indices(task_indices: torch.tensor) -> torch.tensor:
    # if task_indices contains discontinuous integers, transform them to continuous integers
    unique_task_indices = torch.unique(task_indices)
    task_indices_map = {tid.item(): i for i, tid in enumerate(unique_task_indices)}
    return torch.tensor([task_indices_map[tid.item()] for tid in task_indices], device=task_indices.device)

class LeggedRobot(VecTask):
    def __init__(self, cfg: Dict[str, Any], rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """ LeggedRobot env base from
        https://github.com/eureka-research/eurekaverse/blob/main/extreme-parkour/legged_gym/legged_gym/envs/base/legged_robot.py
        """
        self.cfg = AttrDict.from_nested_dicts(cfg)  # this function converts the nested dict into an object class
        self.sim_params = AttrDict.from_nested_dicts(cfg["sim"])  

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_rendering_interval = self.cfg["env"]["cameraRenderingInterval"]

        self.height_samples = None
        self.init_done = False
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )
        self._parse_cfg(self.cfg)

        self.resize_transform = torchvision.transforms.Resize(
            (self.cfg.depth.processed_resolution[1], self.cfg.depth.processed_resolution[0]), 
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )
        
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # randomize initial progress
        self.progress_buf = torch.randint(0, int(self.max_episode_length), (self.num_envs,), device=self.device)
        self.post_physics_step()

    def pre_physics_step(self):
        pass

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = actions.to(self.device)
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_delay_steps) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_delay_steps.pop(0), device=self.device, dtype=torch.float)

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions * self.cfg.control.action_scale, -clip_actions, clip_actions).to(self.device)

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        if self.debug_viz and self.camera_request_visual and self.global_counter % 5 == 0:
            self.render_envs()

        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.depth.use_direction_distillation:
            self.extras["delta_yaw_ok"] = self.delta_yaw < 0.6
        else:
            self.extras["delta_yaw_ok"] = torch.zeros_like(self.delta_yaw).bool()
        self.extras["depth"] = None
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, 0]
        self.extras["inc_goal"] = self.inc_goal

        return {"obs": self.obs_buf, "privileged_obs": self.privileged_obs_buf}, self.rew_buf, self.reset_buf, self.extras

    def get_history_observations(self):
        return self.obs_history_buf
    
    def process_depth_image(self, depth_image, env_id):
        """Process depth image (replicated in ParkourLCMAgent.process_depth())"""
        depth_image = depth_image * -1

        height, width = depth_image.shape
        depth_image = depth_image[self.cfg.depth.crop_top:height-self.cfg.depth.crop_bottom, self.cfg.depth.crop_left:width-self.cfg.depth.crop_right]
        assert depth_image.shape[::-1] == self.cfg.depth.processed_resolution, f"Depth image shape is {depth_image.shape}, expected {self.cfg.depth.processed_resolution}"

        # Replace inf values with valid values
        depth_image = torch.clip(depth_image, -1e6, 1e6)

        # Add random noise (for sim-to-real)
        if np.random.uniform() < self.cfg.depth.blur_prob:
            kernel_size = 5
            blur_transform = torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
            depth_image = blur_transform(depth_image[None, :])[0]
        if np.random.uniform() < self.cfg.depth.erase_prob:
            x = np.random.randint(0, depth_image.shape[1])
            y = np.random.randint(0, depth_image.shape[0])
            h = np.random.randint(*self.cfg.depth.erase_size)
            w = np.random.randint(*self.cfg.depth.erase_size)
            replace_val = np.random.uniform(self.cfg.depth.near_clip, self.cfg.depth.far_clip)
            depth_image = torchvision.transforms.functional.erase(depth_image, x, y, h, w, v=replace_val)

        depth_image += self.cfg.depth.bias_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image += self.cfg.depth.granular_noise * torch.randn_like(depth_image)
        blackout_idxs = torch.where(torch.rand(depth_image.shape, device=depth_image.device) < self.cfg.depth.blackout_noise)
        depth_image[blackout_idxs] = 0.0

        # Clip near and far and normalize
        depth_image = torch.clip(depth_image, self.cfg.depth.near_clip, self.cfg.depth.far_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip) - 0.5

        return depth_image

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.progress_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.depth_buf_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)

        self.gym.end_access_image_tensors(self.sim)

    def _update_goals(self):
        # Delay the goal reach by self.cfg.env.reach_goal_delay seconds
        # self.cfg.env.reach_goal_delay / self.dt is the number of iterations that has passed in that time, and thus
        # we keep incrementing self.reach_goal_timer until it reaches that number

        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        
        self.inc_goal = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[self.inc_goal] += 1
        self.reach_goal_timer[self.inc_goal] = 0
        self.min_dist_to_goal[self.inc_goal] = float('inf')

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.progress_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        
        self._update_goals()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        if (self.common_step_counter-1) % self.camera_rendering_interval == 0:
            self.camera_request_visual = True

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.update_depth_buffer()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self._draw_goals()
            self._draw_feet()
            if self.cfg.depth.use_camera:
                window_name = "Depth (latest, delayed)"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                latest_depth = self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5
                delayed_depth = self.depth_buffer[self.lookat_id, 0].cpu().numpy() + 0.5
                cv2.imshow(window_name, np.concatenate((latest_depth, delayed_depth), axis=0))
                cv2.waitKey(1)

        if self.debug_viz and self.camera_request_visual:
            out = self.get_debug_viz()
            if out is not None:
                for key in out.keys():
                    self.extras[f"debug_visual_{key}"] = out[key]
        else:
            for key in self.extras.keys():
                if "debug_visual" in key:
                    self.extras[key] = None

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        roll_cutoff = torch.abs(self.roll) > 2.5
        pitch_cutoff = torch.abs(self.pitch) > 1.5
        height_cutoff = self.root_states[:, 2] < -0.25
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        if len(self.termination_contact_indices) > 0:
            contact_cutoff = torch.sum(torch.norm(self.contact_forces[:, self.termination_contact_indices], dim=-1), dim=-1) > 2.
        else:
            contact_cutoff = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self.timeout_buf = self.progress_buf > self.max_episode_length # no terminal reward for time-outs
        self.timeout_buf |= reach_goal_cutoff

        self.reset_buf |= self.timeout_buf
        if self.cfg.env.early_termination:
            self.reset_buf |= roll_cutoff
            self.reset_buf |= pitch_cutoff
            self.reset_buf |= height_cutoff
            self.reset_buf |= contact_cutoff

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            raise NotImplementedError
            self._update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        if not self.command_control:
            self._resample_commands(env_ids)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.extras["rew_sums"] = self.rew_sums.clone()
        self.extras["rew_term_sums"] = {name: self.rew_term_sums[name].clone() for name in self.rew_term_sums.keys()}
        self.extras["cur_goal_idx"] = self.cur_goal_idx.clone()

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.obs_history_buf[env_ids, :, :] = 0.
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.reach_goal_timer[env_ids] = 0
        self.progress_buf[env_ids] = 0

        self.min_dist_to_goal[env_ids] = float('inf')

        # fill extras
        self.extras["episode"] = {}
        self.extras["episode"]["rew_total"] = torch.mean(self.rew_sums[env_ids]) / self.max_episode_length_s
        self.extras["episode"]["num_goals_reached"] = self.cur_goal_idx[env_ids].float()

        self.extras["episode"]["success"] = self.cur_goal_idx[env_ids].float() / self.cfg.terrain.num_goals
        if 0 in env_ids:
            print("Rew total", self.extras["episode"]["rew_total"].item(), "Success", torch.mean(self.extras["episode"]["success"]).item())
        for tid in torch.unique(self.extras["task_indices"]):
            mask = self.extras["task_indices"] == tid
            # task_env_ids = mask.nonzero(as_tuple=False).flatten()
            task_env_ids = torch.logical_and(mask, ~self.reset_buf).nonzero(as_tuple=False).flatten()
            self.extras["episode"][f"num_goals_reached_{tid.item()}"] = torch.mean(self.cur_goal_idx[task_env_ids].float())
        self.rew_sums[env_ids] = 0.
        for key in self.rew_term_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.rew_term_sums[key][env_ids]) / self.max_episode_length_s
            self.rew_term_sums[key][env_ids] = 0.

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            self.extras["episode"]["highest_terrain_level"] = torch.mean(self.highest_terrain_levels.float())
            self.extras["episode"]["randomize_level"] = torch.mean(self.randomize_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.timeout_buf

        self.cur_goal_idx[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]

            self.rew_buf += rew                                              # Tracks reward sum for current step (from BaseTask)
            self.rew_sums += rew                                             # Tracks reward sum for current episode, summed over steps
            self.rew_term_sums[name] += rew                                  # Tracks reward terms for current episode, summed over steps

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.rew_term_sums["termination"] += rew
    
    def compute_observations(self):
        """ 
        Computes observations
        """
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        if self.global_counter % 5 == 0:
            self.delta_yaw = self.target_yaw - self.yaw
            self.delta_next_yaw = self.next_target_yaw - self.yaw

        # NOTE: This is proprioception and a few other inputs, but we call it proprioception for simplicity
        proprio = torch.cat((
            self.base_ang_vel  * self.obs_scales.ang_vel,
            imu_obs,
            self.delta_yaw[:, None],
            self.delta_next_yaw[:, None],
            self.commands[:, 0:1],
            (self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.action_history_buf[:, -1],
            self.contact_filt.float() - 0.5,
        ), dim=-1)
        assert proprio.shape[1] == self.cfg.env.n_proprio

        priv_explicit = self.base_lin_vel * self.obs_scales.lin_vel
        priv_latent = torch.cat((
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1, 
            self.motor_strength[1] - 1
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            self.obs_buf = torch.cat([proprio, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1), self.task_embedding], dim=-1)
        else:
            self.obs_buf = torch.cat([proprio, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1), self.task_embedding], dim=-1)

        # Mask yaw in proprioceptive history
        proprio[:, 5:7] = 0
        if self.cfg.env.history_len > 0:
            self.obs_history_buf = torch.where(
                (self.progress_buf <= 1)[:, None, None], 
                torch.stack([proprio] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.obs_history_buf[:, 1:],
                    proprio.unsqueeze(1)
                ], dim=1)
            )

        self.contact_buf = torch.where(
            (self.progress_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )
        
        
    def get_noisy_measurement(self, x, scale):
        if self.cfg.noise.add_noise:
            x = x + (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
        return x

    def create_sim(self):
        """ Creates simulation, terrain and environments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        start = time()
        print("*"*80)
        print("Start creating ground...")
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.progress_buf % int(self.cfg.commands.resampling_time / self.dt)==0)
        if self.command_control:
            # User is setting commands via WASD keys, don't overwrite
            # Instead, just make sure command values are within range
            self._clip_commands()
        else:
            self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())

        # If heading command is used, need to set ang_vel_yaw command as heading error
        if "heading" in self.cfg.commands.commands and "ang_vel_yaw" in self.cfg.commands.commands:
            heading_idx = self.cfg.commands.commands.index("heading")
            ang_vel_yaw_idx = self.cfg.commands.commands.index("ang_vel_yaw")
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, ang_vel_yaw_idx] = torch.clip(0.8*wrap_to_pi(self.commands[:, heading_idx] - heading), -1., 1.)
            self.commands[:, ang_vel_yaw_idx] *= torch.abs(self.commands[:, ang_vel_yaw_idx]) > self.cfg.commands.ang_vel_clip
        
        if self.cfg.terrain.measure_heights:
            if self.global_counter % self.cfg.depth.update_interval == 0:
                self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        
    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]+future).expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)
    
    def _clip_commands(self):
        old_lookat_speed = self.commands[self.lookat_id, 0].item()
        for command_name in self.cfg.commands.commands:
            idx = self.cfg.commands.commands.index(command_name)
            self.commands[:, idx] = torch.clip(self.commands[:, idx], self.command_ranges[command_name][0], self.command_ranges[command_name][1])
            # Set small velocity commands to zero
            if command_name == "lin_vel_x" or command_name == "lin_vel_y":
                self.commands[:, idx] *= torch.abs(self.commands[:, idx]) >= self.cfg.commands.lin_vel_clip

        if self.commands[self.lookat_id, 0] != old_lookat_speed:
            print(f"Commanded speed clipped to {self.commands[self.lookat_id, 0]}")

    def _resample_commands(self, env_ids):
        for command_name in self.cfg.commands.commands:
            if command_name == "ang_vel_yaw" and "heading" in self.cfg.commands.commands:
                # If heading command is used, ang_vel_yaw is set as heading error in _post_physics_step_callback()
                continue

            idx = self.cfg.commands.commands.index(command_name)
            self.commands[env_ids, idx] = torch_rand_float(self.command_ranges[command_name][0], self.command_ranges[command_name][1], (len(env_ids), 1), device=self.device).squeeze(1)
            # Set small velocity commands to zero
            if command_name == "lin_vel_x" or command_name == "lin_vel_y":
                self.commands[env_ids, idx] *= torch.abs(self.commands[env_ids, idx]) >= self.cfg.commands.lin_vel_clip
            if command_name == "lin_vel_x" and not torch.all(self.env_class == -1) and self.cfg.terrain.curriculum:
                # If we're training on any non-flat terrains, we should not command 0 speed because it disrupts the curriculum
                assert self.command_ranges[command_name][0] >= self.cfg.commands.lin_vel_clip, "Minimum speed command should be greater than 0 when training on non-flat terrains"

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """

        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        if self.cfg.domain_rand.action_delay:
            indices = -1 - self.delay * self.cfg.control.decimation
            actions = self.action_history_buf[:, indices.long()]

        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_motor:
                torques = self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains*self.dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[1] * self.d_gains*self.dof_vel
                
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.9, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0*rand_yaw, rand_pitch, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]  
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        
        if self.cfg.terrain.type == "original" or self.cfg.terrain.type == "original_distill":
            # Distance-based curriculum, used with original terrain
            dis_to_origin = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
            threshold = self.commands[env_ids, 0] * self.cfg.env.episode_length_s
            move_up = dis_to_origin > 0.8 * threshold
            move_down = dis_to_origin < 0.4 * threshold
        else:
            # Goal-based curriculum, based solely on goal progression rather than distance
            # Hard variant, focuses on pareto front
            # move_up = self.cur_goal_idx[env_ids] >= 1.0 * self.cfg.terrain.num_goals
            # move_down = self.cur_goal_idx[env_ids] < 0.125 * self.cfg.terrain.num_goals

            # Soft variant, diversifies levels
            move_up = self.cur_goal_idx[env_ids] >= 0.8 * self.cfg.terrain.num_goals
            move_down = self.cur_goal_idx[env_ids] < 0.4 * self.cfg.terrain.num_goals
            no_move = ~(move_up | move_down)
            randomize_no_move = torch.rand_like(no_move.float()) < 0.25

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], min=0)
        self.highest_terrain_levels = torch.maximum(self.highest_terrain_levels, self.terrain_levels)
        # Agents that pass last level are sent to random previous level
        self.randomize_levels = self.terrain_levels[env_ids] >= self.max_terrain_level
        # In soft variant, some agents that are stuck on current level are also sent to random previous level
        self.randomize_levels = self.randomize_levels | (no_move & randomize_no_move)
        random_level = (torch.rand_like(self.terrain_levels[env_ids].float()) * self.terrain_levels[env_ids]).to(torch.long)
        assert torch.max(random_level) < self.max_terrain_level, "Random level exceeds max level!"
        assert torch.all(random_level <= self.terrain_levels[env_ids]), "Random level exceeds current level!"
        self.terrain_levels[env_ids] = torch.where(self.randomize_levels, random_level, self.terrain_levels[env_ids])
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
            
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6) # for feet only, see create_env()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        terrain_levels = torch.arange(self.cfg.terrain.num_rows, device=self.device).repeat(self.num_envs // self.cfg.terrain.num_rows + 1)[:self.num_envs].to(torch.long)
        self.extras["ordered_task_names"] = self.extras["task_indices"] = self.env_class  # transform_task_indices(self.env_class) # * self.cfg.terrain.num_rows + terrain_levels
        # self.extras["ordered_task_names"] = self.extras["task_indices"] = torch.randint(0, 21, (self.num_envs,), device=self.device)
        self.task_embedding = torch.nn.functional.one_hot(transform_task_indices(self.extras["task_indices"]), 21).float().view(self.num_envs, -1)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.inc_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 4, device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, len(self.cfg.commands.commands), dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self._resample_commands(torch.arange(self.num_envs, device=self.device, requires_grad=False))
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.command_control = False

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_dof_pos_all[:] = self.default_dof_pos[0]

        self.height_update_interval = 1
        if hasattr(self.cfg.env, "height_update_dt"):
            self.height_update_interval = int(self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.depth_buf_len,
                                            self.cfg.depth.processed_resolution[1], 
                                            self.cfg.depth.processed_resolution[0]).to(self.device)
        
        self.privileged_obs_buf = None

        if self.debug_viz:
            self.camera_frames = {
                "main": [],
                "side": [],
                "top": []
            }

        self.camera_request_visual = False

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        self.min_dist_to_goal = torch.tensor([float('inf') for _ in range(self.num_envs)], dtype=torch.float, device=self.device, requires_grad=False)

        # rewards in current episode
        self.rew_sums = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.rew_term_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border
        hf_params.transform.p.y = -self.terrain.border
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.flatten(order='C'), hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            print("Attaching camera to actor")
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width, camera_props.height = self.cfg.depth.original_resolution
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov 
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)
            
            local_transform = gymapi.Transform()
            
            camera_position = np.random.normal(config.position["mean"], config.position["std"])
            camera_rotation = np.random.normal(config.rotation["mean"], config.rotation["std"])
            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(*camera_rotation)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
            
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def get_debug_viz(self):
        if not self.debug_viz or not self.camera_request_visual:
            return None

        if self.camera_request_visual:
            if len(self.camera_frames["main"]) < self.max_episode_length / 5:
                # print(self.max_episode_length / 5 - len(self.camera_frames["main"]))
                return None
            else:
                print("End camera rendering")
                self.camera_request_visual = False
                camera_frames = {}
                for key in self.camera_frames.keys():
                    for i in range(len(self.render_env_indices)):
                        camera_frames_tmp = torch.stack([frames[i] for frames in self.camera_frames[key]], dim=0).permute(0, 3, 1, 2).unsqueeze(0)
                        camera_frames[key + f"_{self.render_env_indices[i]}"] = camera_frames_tmp
                self.camera_frames = {"main": [], "side": [], "top": []}
                return camera_frames
        else:
            return None

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../../assets/urdf/go1/robots/go1_with_camera.urdf")
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        for s in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        print("Creating envs...")
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.env.rand_yaw_range*np.random.uniform(-1, 1))
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)
            
            self.attach_camera(i, env_handle, anymal_handle)

            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        thigh_names = ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_names):
            self.thigh_indices[i] = self.dof_names.index(name)
        calf_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(calf_names):
            self.calf_indices[i] = self.dof_names.index(name)
        
        if self.debug_viz:
            # Set up camera for each env, used to visualize terrain
            # NOTE: A viewer might seem more suited for this, but IsaacGym links the viewer with a GUI window, which seems unavoidable
            terrain_length, terrain_width, eps = self.cfg.terrain.terrain_length, self.cfg.terrain.terrain_width, 1e-3
            self.env_cameras = {"main": [], "side": [], "top": []}
            self.env_camera_location_viewpoint_offsets = {
                "main": ([terrain_length // 2, terrain_width + 4, 7], [terrain_length // 2, terrain_width // 2, 0]),
                "side": ([terrain_length // 2, terrain_width + 6, 2], [terrain_length // 2, terrain_width // 2, 0]),
                "top": ([terrain_length // 2, terrain_width // 2 + eps, 9], [terrain_length // 2, terrain_width // 2, 0]),
                # Note: eps needed to avoid isaacgym rendering bug (black image) when looking straight down
            }
            # get unique env indices (all unique terrain categories)
            # _, self.render_env_indices = np.unique((self.env_class * self.cfg.terrain.num_rows + self.terrain_levels).cpu().detach().numpy(), return_index=True)
            # get specific env indices to render
            self.render_env_indices = self.cfg.env.render_env_indices
            print(f"Setting up debug viz cameras in envs {self.render_env_indices}...")
            self.camera_tensor = {
                "main": [],
                "side": [],
                "top": []
            }
            for env_id in self.render_env_indices:
                env_camera_props = gymapi.CameraProperties()
                env_camera_props.width = 720
                env_camera_props.height = 240
                env_camera_props.enable_tensors = True
                for viewpoint_name, (cam_position_offset, cam_target_offset) in self.env_camera_location_viewpoint_offsets.items():
                    env_camera = self.gym.create_camera_sensor(self.envs[env_id], env_camera_props)

                    # Get coordinates of the environment
                    cam_position = [self.terrain_levels[env_id] * terrain_length, self.terrain_types[env_id] * terrain_width, 0]
                    cam_target = [self.terrain_levels[env_id] * terrain_length, self.terrain_types[env_id] * terrain_width, 0]
                    # Add camera offset to the environment location
                    cam_position = [cam_position[i] + cam_position_offset[i] for i in range(3)]
                    cam_target = [cam_target[i] + cam_target_offset[i] for i in range(3)]

                    self.gym.set_camera_location(env_camera, self.envs[env_id], gymapi.Vec3(*cam_position), gymapi.Vec3(*cam_target))
                    self.env_cameras[viewpoint_name].append(env_camera)

                    # obtain camera tensors
                    video_frame_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_id], env_camera, gymapi.IMAGE_COLOR)
                    self.camera_tensor[viewpoint_name].append(gymtorch.wrap_tensor(video_frame_tensor).view((env_camera_props.height, env_camera_props.width, 4)))
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            # put robots at the origins defined by the terrain
            self.max_terrain_level = self.cfg.terrain.num_rows
            max_init_level = min(self.cfg.terrain.max_init_terrain_level, self.cfg.terrain.num_rows - 1)
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            if not self.cfg.terrain.curriculum:
                # Evenly distribute across levels
                max_init_level = self.cfg.terrain.num_rows - 1
                self.terrain_levels = torch.arange(self.cfg.terrain.num_rows, device=self.device).repeat(self.num_envs // self.cfg.terrain.num_rows + 1)[:self.num_envs].to(torch.long)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.highest_terrain_levels = self.terrain_levels.clone()  # Saves the maximum level reached by each robot (because they can go back to lower levels)
            self.randomize_levels = torch.zeros_like(self.terrain_levels, dtype=torch.bool, device=self.device, requires_grad=False)
            # terrain_levels is the difficulty levels of the terrain
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
            self.cur_goals = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
            self.env_class = torch.ones(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long) * -1
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)

            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
            self.cur_goals = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = self.cfg.rewards.scales
        reward_norm_factor = 1  # np.sum(list(self.reward_scales.values()))
        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor
        if self.cfg.commands.curriculum:
            self.command_ranges = self.cfg.commands.curriculum_ranges
        else:
            self.command_ranges = self.cfg.commands.ranges
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_height_samples(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        i = self.lookat_id
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        for j in range(heights.shape[0]):
            x = height_points[j, 0] + base_pos[0]
            y = height_points[j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    def _draw_goals(self):
        if not self.cfg.depth.use_camera:
            # Only for scandot poliices, since wireframe shows up on depth camera for some reason

            sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))
            sphere_geom_cur = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 0, 1))
            sphere_geom_reached = gymutil.WireframeSphereGeometry(self.cfg.env.next_goal_threshold, 32, 32, None, color=(0, 1, 0))
            goals = self.terrain_goals[self.terrain_levels[self.lookat_id], self.terrain_types[self.lookat_id]].cpu().numpy()
            for i, goal in enumerate(goals):
                goal_xy = goal[:2] + self.terrain.cfg.border_size
                pts = (goal_xy/self.terrain.cfg.horizontal_scale).astype(int)
                if pts[0] < 0 or pts[0] >= self.terrain.tot_rows or pts[1] < 0 or pts[1] >= self.terrain.tot_cols:
                    print("Goal out of bounds!")
                    continue
                goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
                pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
                if i == self.cur_goal_idx[self.lookat_id].cpu().item():
                    gymutil.draw_lines(sphere_geom_cur, self.gym, self.viewer, self.envs[self.lookat_id], pose)
                    if self.reached_goal_ids[self.lookat_id]:
                        gymutil.draw_lines(sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)
                else:
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

            norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
            target_vec_norm = self.target_pos_rel / (norm + 1e-5)
            next_norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
            next_target_vec_norm = self.next_target_pos_rel / (next_norm + 1e-5)

            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
            for i in range(5):
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
            for i in range(5):
                pose_arrow = pose_robot[:2] + 0.2*(i+3) * next_target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
    def _draw_feet(self):
        if hasattr(self, 'feet_at_edge'):
            non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
            edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0, 0))

            feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
            for i in range(4):
                pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, i, 0], feet_pos[self.lookat_id, i, 1], feet_pos[self.lookat_id, i, 2]), r=None)
                if self.feet_at_edge[self.lookat_id, i]:
                    gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
                else:
                    gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points

    def get_foot_contacts(self):
        foot_contacts_bool = self.contact_forces[:, self.feet_indices, 2] > 10
        if self.cfg.env.include_foot_contacts:
            return foot_contacts_bool
        else:
            return torch.zeros_like(foot_contacts_bool).to(self.device)

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_heights_points(self, coords, env_ids=None):
        if env_ids:
            points = coords[env_ids]
        else:
            points = coords

        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def render_envs(self):
        self.gym.step_graphics(self.sim)
        # render sensors and refresh camera tensors
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        # stack across all envs
        # self.camera_tensor[VIEWPOINT] in shape list of num_envs tuple (height, width, 4)
        self.camera_frames["main"].append(torch.stack(self.camera_tensor["main"], dim=0).clone())
        self.camera_frames["side"].append(torch.stack(self.camera_tensor["side"], dim=0).clone())
        self.camera_frames["top"].append(torch.stack(self.camera_tensor["top"], dim=0).clone())


        self.gym.end_access_image_tensors(self.sim)

    ################## parkour rewards ##################

    def _reward_tracking_goal_vel(self):
        target_vel = self.target_pos_rel / (torch.norm(self.target_pos_rel, dim=-1, keepdim=True) + 1e-5)
        # print(self.cur_goal_idx[0], self.target_pos_rel[0])
        cur_vel = self.root_states[:, 7:9]
        proj_vel = torch.sum(target_vel * cur_vel, dim=-1)
        command_vel = self.commands[:, 0]

        # This rewards velocity up to the command velocity, then plateaus
        # We use this for positive velocity since some obstacles may require more
        # than the commanded speed to pass
        rew_move = torch.minimum(proj_vel, command_vel) / (command_vel + 1e-5)
        # This rewards is maximum at the command velocity and forms a Gaussian around it
        # We use this for zero velocity to teach the robot to stop
        rew_still = torch.exp(-torch.square(proj_vel - command_vel) / 0.2)

        rew = torch.zeros_like(proj_vel)
        rew[self.commands[:, 0] > 0] = rew_move[self.commands[:, 0] > 0]
        rew[self.commands[:, 0] == 0] = rew_still[self.commands[:, 0] == 0]

        return rew

    def _reward_negative_vel_penalty(self):
        target_vel = self.target_pos_rel / (torch.norm(self.target_pos_rel, dim=-1, keepdim=True) + 1e-5)
        cur_vel = self.root_states[:, 7:9]
        proj_vel = torch.sum(target_vel * cur_vel, dim=-1)
        command_vel = self.commands[:, 0]
        return (proj_vel <= 0.05) * (command_vel > 0)

    def _reward_exploration(self):
        EPSILON = 1e-5
        r = torch.sum(
            self.base_lin_vel[:, :2] * \
            self.target_pos_rel[:, :2]
        , dim=1)
        r /= (torch.norm(self.target_pos_rel[:, :2], dim=1) + EPSILON)
        r /= (torch.norm(self.base_lin_vel[:, :2], dim=1) + EPSILON)
        r *= (torch.norm(self.base_lin_vel[:, :2], dim=1) > 0.2).float()  # only reward when moving fast enough
        return r

    def _reward_reach_goal(self):
        return self.reached_goal_ids.float()

    def _reward_tracking_yaw(self):
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        return rew
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        rew[self.env_class != -1] *= 0.5  # Only for flat terrain
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
     
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        rew[self.env_class != -1] = 0.0  # Only for flat terrain
        return rew

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    
    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew
    
    def _reward_energy_cost(self):
        energy_cost = torch.maximum(self.torques * self.dof_vel, torch.zeros_like(self.torques))
        return torch.sum(energy_cost, dim=1)