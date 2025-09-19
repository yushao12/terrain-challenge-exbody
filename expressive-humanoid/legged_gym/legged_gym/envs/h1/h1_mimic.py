from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision

from legged_gym import LEGGED_GYM_ROOT_DIR, ASE_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from legged_gym.utils.math import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

import sys
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/challenging_terrain')
from terrain_base.terrain import Terrain as ChallengingTerrain
from terrain_base.config import terrain_config

import sys
sys.path.append(os.path.join(ASE_DIR, "ase"))
sys.path.append(os.path.join(ASE_DIR, "ase/utils"))
import cv2

from motion_lib import MotionLib
import torch_utils

class H1Mimic(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        
        # 启用challenging terrain
        self.use_challenging_terrain = True
        
        # ===== ASE风格的Location Goals配置 =====
        # 是否使用ASE风格的动态location goals（替代terrain goals）
        self.use_ase_location_goals = False
        
        # ASE location goals参数
        self._tar_speed = getattr(cfg.env, 'tar_speed', 1.0)  # 目标速度
        self._tar_change_steps_min = getattr(cfg.env, 'tar_change_steps_min', 100)  # 目标变化最小步数
        self._tar_change_steps_max = getattr(cfg.env, 'tar_change_steps_max', 300)  # 目标变化最大步数
        self._tar_dist_max = getattr(cfg.env, 'tar_dist_max', 3.0)  # 目标最大距离
        self._goal_reach_threshold = getattr(cfg.env, 'goal_reach_threshold', 0.5)  # 到达目标阈值
        
        # Pre init for motion loading
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        
        self.init_motions(cfg)
        if cfg.motion.num_envs_as_motions:
            self.cfg.env.num_envs = self._motion_lib.num_motions()
        
        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        
        # ===== 初始化ASE Location Goals缓冲区 =====
        if self.use_ase_location_goals:
            self._init_ase_location_goals()
        
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

        self.init_motion_buffers(cfg)
        # self.rand_vx_cmd = 4*torch.rand((self.num_envs, ), device=self.device) - 2

        self.reset_idx(torch.arange(self.num_envs, device=self.device), init=True)
        # 移除过早的post_physics_step调用，避免在初始化时访问未初始化的缓冲区
        self.post_physics_step()

    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        noise_scale_vec[:, :3] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, 3:5] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, 7:7+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, 7+self.num_dof:7+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        return noise_scale_vec
    
    def init_motions(self, cfg):
        self._key_body_ids = torch.tensor([3, 6, 9, 12], device=self.device)  #self._build_key_body_ids_tensor(key_bodies)
        # ['pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 
        # 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 
        # 'torso_link', 
        # 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_hand_keypoint_link', 
        # 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_hand_keypoint_link']
        self._key_body_ids_sim = torch.tensor([1, 4, 5, # Left Hip yaw, Knee, Ankle
                                               6, 9, 10,
                                               12, 15, 16, # Left Shoulder pitch, Elbow, hand
                                               17, 20, 21], device=self.device)
        self._key_body_ids_sim_subset = torch.tensor([6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
        
        self._num_key_bodies = len(self._key_body_ids_sim_subset)
        self._dof_body_ids = [1, 2, 3, # Hip, Knee, Ankle
                              4, 5, 6,
                              7,       # Torso
                              8, 9, 10, # Shoulder, Elbow, Hand
                              11, 12, 13]  # 13
        self._dof_offsets = [0, 3, 4, 5, 8, 9, 10, 
                             11, 
                             14, 15, 16, 19, 20, 21]  # 14
        self._valid_dof_body_ids = torch.ones(len(self._dof_body_ids)+2*4, device=self.device, dtype=torch.bool)
        self._valid_dof_body_ids[-1] = 0
        self._valid_dof_body_ids[-6] = 0
        self.dof_indices_sim = torch.tensor([0, 1, 2, 5, 6, 7, 11, 12, 13, 16, 17, 18], device=self.device, dtype=torch.long)
        self.dof_indices_motion = torch.tensor([2, 0, 1, 7, 5, 6, 12, 11, 13, 17, 16, 18], device=self.device, dtype=torch.long)
        
        # self._dof_ids_subset = torch.tensor([0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device)  # no knee and ankle
        self._dof_ids_subset = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device)  # no knee and ankle
        self._n_demo_dof = len(self._dof_ids_subset)

        #['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 
        #'left_knee_joint', 'left_ankle_joint', 
        #'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 
        #'right_knee_joint', 'right_ankle_joint', 
        #'torso_joint', 
        #'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
        #'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
        # self.dof_ids_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device, dtype=torch.long)
        # motion_name = "17_04_stealth"
        if cfg.motion.motion_type == "single":
            motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/retarget_npy/{cfg.motion.motion_name}.npy")
        else:
            assert cfg.motion.motion_type == "yaml"
            motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/configs/{cfg.motion.motion_name}")
        
        self._load_motion(motion_file, cfg.motion.no_keybody)

    def init_motion_buffers(self, cfg):
        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)
        if cfg.motion.motion_curriculum:
            self._max_motion_difficulty = 9
            # self._motion_ids = self._motion_lib.sample_motions(self.num_envs, self._max_motion_difficulty)
        else:
            self._max_motion_difficulty = 9
        self._motion_times = self._motion_lib.sample_time(self._motion_ids)
        self._motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        self._motion_difficulty = self._motion_lib.get_motion_difficulty(self._motion_ids)
        # self._motion_features = self._motion_lib.get_motion_features(self._motion_ids)

        self._motion_dt = self.dt
        self._motion_num_future_steps = self.cfg.env.n_demo_steps
        self._motion_demo_offsets = torch.arange(0, self.cfg.env.n_demo_steps * self.cfg.env.interval_demo_steps, self.cfg.env.interval_demo_steps, device=self.device)
        self._demo_obs_buf = torch.zeros((self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo), device=self.device)
        self._curr_demo_obs_buf = self._demo_obs_buf[:, 0, :]
        self._next_demo_obs_buf = self._demo_obs_buf[:, 1, :]
        # self._curr_mimic_obs_buf = torch.zeros_like(self._curr_demo_obs_buf, device=self.device)

        self._curr_demo_root_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._curr_demo_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._curr_demo_root_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._curr_demo_keybody = torch.zeros((self.num_envs, self._num_key_bodies, 3), device=self.device)
        self._in_place_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.dof_term_threshold = 3 * torch.ones(self.num_envs, device=self.device)
        self.keybody_term_threshold = 0.3 * torch.ones(self.num_envs, device=self.device)
        self.yaw_term_threshold = 0.5 * torch.ones(self.num_envs, device=self.device)
        self.height_term_threshold = 0.2 * torch.ones(self.num_envs, device=self.device)

        # self.step_inplace_ids = self.resample_step_inplace_ids()
    
    def _load_motion(self, motion_file, no_keybody=False):
        # assert(self._dof_offsets[-1] == self.num_dof + 2)  # +2 for hand dof not used
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device, 
                                     no_keybody=no_keybody, 
                                     regen_pkl=self.cfg.motion.regen_pkl)
        return
    
    def step(self, actions):
        actions = self.reindex(actions)

        actions.to(self.device)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            # self.delay = torch.randint(0, 3, (1,), device=self.device, dtype=torch.float)
            indices = -self.delay -1
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
                
        self.actions[:, [4, 9]] = torch.clamp(self.actions[:, [4, 9]], -0.5, 0.5)
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # for i in torch.topk(self.torques[self.lookat_id], 3).indices.tolist():
        #     print(self.dof_names[i], self.torques[self.lookat_id][i])
        
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def resample_motion_times(self, env_ids):
        return self._motion_lib.sample_time(self._motion_ids[env_ids])
    
    def update_motion_ids(self, env_ids):
        # 调试信息：检查motion_ids范围
        motion_ids_to_update = self._motion_ids[env_ids]
        max_motion_id = motion_ids_to_update.max().item()
        min_motion_id = motion_ids_to_update.min().item()
        num_motions_in_lib = self._motion_lib.num_motions()
        
        print(f"[DEBUG] update_motion_ids: env_ids={len(env_ids)}, motion_ids range=[{min_motion_id}, {max_motion_id}], lib has {num_motions_in_lib} motions")
        
        # 检查是否有越界的motion_ids
        if max_motion_id >= num_motions_in_lib:
            print(f"[ERROR] Motion ID {max_motion_id} >= num_motions {num_motions_in_lib}")
            print(f"[ERROR] Problematic motion_ids: {motion_ids_to_update[motion_ids_to_update >= num_motions_in_lib]}")
            # 修复越界的motion_ids
            motion_ids_to_update = torch.remainder(motion_ids_to_update, num_motions_in_lib)
            self._motion_ids[env_ids] = motion_ids_to_update
            print(f"[FIXED] Corrected motion_ids range=[{motion_ids_to_update.min().item()}, {motion_ids_to_update.max().item()}]")
        
        try:
            # 检查特定的motion_ids是否总是出现（这些可能是问题motion）
            suspicious_motion_ids = [1, 2, 10, 12, 653, 655, 668, 676]
            for motion_id in suspicious_motion_ids:
                if motion_id in motion_ids_to_update:
                    print(f"[WARNING] Suspicious motion_id {motion_id} detected in update!")
                    # 检查这个motion的数据是否正常
                    try:
                        test_length = self._motion_lib.get_motion_length(torch.tensor([motion_id], device=self.device))
                        test_time = self._motion_lib.sample_time(torch.tensor([motion_id], device=self.device))
                        print(f"[DEBUG] Motion {motion_id}: length={test_length.item()}, time={test_time.item()}")
                    except Exception as e:
                        print(f"[ERROR] Motion {motion_id} has corrupted data: {e}")
                        # 替换这个有问题的motion_id
                        mask = motion_ids_to_update == motion_id
                        if mask.any():
                            replacement_id = torch.randint(0, num_motions_in_lib, (mask.sum(),), device=self.device)
                            motion_ids_to_update[mask] = replacement_id
                            self._motion_ids[env_ids] = motion_ids_to_update
                            print(f"[FIXED] Replaced problematic motion_id {motion_id} with {replacement_id.tolist()}")
            
            self._motion_times[env_ids] = self.resample_motion_times(env_ids)
            self._motion_lengths[env_ids] = self._motion_lib.get_motion_length(self._motion_ids[env_ids])
            self._motion_difficulty[env_ids] = self._motion_lib.get_motion_difficulty(self._motion_ids[env_ids])
            
            # 验证更新后的张量形状
            if self._motion_times.shape != self._motion_lengths.shape:
                print(f"[ERROR] Shape mismatch after update: _motion_times {self._motion_times.shape} vs _motion_lengths {self._motion_lengths.shape}")
                # 重新初始化整个motion系统
                self._motion_times = self._motion_lib.sample_time(self._motion_ids)
                self._motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
                self._motion_difficulty = self._motion_lib.get_motion_difficulty(self._motion_ids)
                print(f"[DEBUG] Reinitialized entire motion system")
                
        except Exception as e:
            print(f"[ERROR] Failed to update motion_ids: {e}")
            # 完全重新初始化motion系统
            self._motion_times = self._motion_lib.sample_time(self._motion_ids)
            self._motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
            self._motion_difficulty = self._motion_lib.get_motion_difficulty(self._motion_ids)
            print(f"[DEBUG] Completely reinitialized motion system due to error")

    def _init_ase_location_goals(self):
        """初始化ASE风格的location goals缓冲区"""
        # 目标变化步数计数器
        self._tar_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        # 上一帧的机器人位置
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        # 当前目标位置 (x, y)
        self._tar_pos = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        # 目标到达计数器
        self._goal_reach_timer = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        
        # 初始化ASE goal朝向相关变量
        self.ase_goal_yaw = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self.ase_delta_yaw = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        
        # 初始化目标位置
        self._reset_ase_location_goals(torch.arange(self.num_envs, device=self.device))

    def _reset_ase_location_goals(self, env_ids):
        """重置ASE风格的location goals"""
        if len(env_ids) == 0:
            return
            
        n = len(env_ids)
        
        # 获取机器人当前位置
        char_root_pos = self.root_states[env_ids, 0:2]
        
        # 生成随机目标位置（在机器人周围）
        rand_pos = self._tar_dist_max * (2.0 * torch.rand([n, 2], device=self.device) - 1.0)
        
        # 设置目标位置
        self._tar_pos[env_ids] = char_root_pos + rand_pos
        
        # 设置目标变化步数
        change_steps = torch.randint(
            low=self._tar_change_steps_min, 
            high=self._tar_change_steps_max,
            size=(n,), 
            device=self.device, 
            dtype=torch.int64
        )
        self._tar_change_steps[env_ids] = self.episode_length_buf[env_ids] + change_steps
        
        # 重置目标到达计时器
        self._goal_reach_timer[env_ids] = 0.0

    def _init_buffers(self):
        """重写基类的_init_buffers方法，确保obs_history_buf使用正确的n_proprio维度"""
        # 调用基类的_init_buffers方法
        super()._init_buffers()
        
        # 重新初始化obs_history_buf，使用当前项目的n_proprio维度
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(
                self.num_envs, 
                self.cfg.env.history_len, 
                self.cfg.env.n_proprio,  # 使用当前项目的n_proprio
                device=self.device, 
                dtype=torch.float
            )

    def reset_idx(self, env_ids, init=False):
        if len(env_ids) == 0:
            return
        
        print(f"[DEBUG] reset_idx called with {len(env_ids)} environments, init={init}")
        # RSI
        if self.cfg.motion.motion_curriculum:
            # ep_length = self.episode_length_buf[env_ids] * self.dt
            completion_rate = self.episode_length_buf[env_ids] * self.dt / self._motion_lengths[env_ids]
            completion_rate_mean = completion_rate.mean()
            # if completion_rate_mean > 0.8:
            #     self._max_motion_difficulty = min(self._max_motion_difficulty + 1, 9)
            #     self._motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids), self._max_motion_difficulty)
            # elif completion_rate_mean < 0.4:
            #     self._max_motion_difficulty = max(self._max_motion_difficulty - 1, 0)
            #     self._motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids), self._max_motion_difficulty)
            relax_ids = completion_rate < 0.3
            strict_ids = completion_rate > 0.9
            # self.dof_term_threshold[env_ids[relax_ids]] += 0.05
            self.dof_term_threshold[env_ids[strict_ids]] -= 0.05
            self.dof_term_threshold.clamp_(1.5, 3)

            self.height_term_threshold[env_ids[relax_ids]] += 0.01
            self.height_term_threshold[env_ids[strict_ids]] -= 0.01
            self.height_term_threshold.clamp_(0.03, 0.1)

            relax_ids = completion_rate < 0.6
            strict_ids = completion_rate > 0.9
            self.keybody_term_threshold[env_ids[relax_ids]] -= 0.05
            self.keybody_term_threshold[env_ids[strict_ids]] += 0.05
            self.keybody_term_threshold.clamp_(0.1, 0.4)

            relax_ids = completion_rate < 0.4
            strict_ids = completion_rate > 0.8
            self.yaw_term_threshold[env_ids[relax_ids]] -= 0.05
            self.yaw_term_threshold[env_ids[strict_ids]] += 0.05
            self.yaw_term_threshold.clamp_(0.1, 0.6)


        self.update_motion_ids(env_ids)

        motion_ids = self._motion_ids[env_ids]
        motion_times = self._motion_times[env_ids]
        root_pos, root_rot, dof_pos_motion, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        # Intialize dof state from default position and reference position
        dof_pos_motion, dof_vel = self.reindex_dof_pos_vel(dof_pos_motion, dof_vel)

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids, dof_pos_motion, dof_vel)
        self._reset_root_states(env_ids, root_vel, root_rot, root_pos[:, 2])

        if init:
            self.init_root_pos_global = self.root_states[:, :3].clone()
            self.init_root_pos_global_demo = root_pos[:].clone()
            self.target_pos_abs = self.init_root_pos_global.clone()[:, :2]
        else:
            self.init_root_pos_global[env_ids] = self.root_states[env_ids, :3].clone()
            self.init_root_pos_global_demo[env_ids] = root_pos[:].clone()
            self.target_pos_abs[env_ids] = self.init_root_pos_global[env_ids].clone()[:, :2]

        self._resample_commands(env_ids)  # no resample commands
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        # 重置接触状态，防止CUDA索引越界
        self.last_contacts[env_ids] = 0.
        # ===== 注释掉原有的terrain goals重置逻辑 =====
        # self.cur_goal_idx[env_ids] = 0
        # self.reach_goal_timer[env_ids] = 0
        
        # ===== 重置ASE Location Goals =====
        if self.use_ase_location_goals:
            self._reset_ase_location_goals(env_ids)
            # 更新上一帧位置
            self._prev_root_pos[env_ids] = self.root_states[env_ids, :3]
        else:
            # 保留原有的terrain goals逻辑
            self.cur_goal_idx[env_ids] = 0
            self.reach_goal_timer[env_ids] = 0

        # 初始化/重置目标达成追踪计时（用于“未达下一个点”早停）
        if not hasattr(self, '_last_goal_idx'):
            self._last_goal_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if not hasattr(self, '_last_goal_step'):
            self._last_goal_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # fill extras
        self.extras["episode"] = {}
        self.extras["episode"]["curriculum_completion"] = completion_rate_mean
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        self.extras["episode"]["curriculum_motion_difficulty_level"] = self._max_motion_difficulty
        self.extras["episode"]["curriculum_dof_term_thresh"] = self.dof_term_threshold.mean()
        self.extras["episode"]["curriculum_keybody_term_thresh"] = self.keybody_term_threshold.mean()
        self.extras["episode"]["curriculum_yaw_term_thresh"] = self.yaw_term_threshold.mean()
        self.extras["episode"]["curriculum_height_term_thresh"] = self.height_term_threshold.mean()
        
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        return
                                                                                                                                                                                                                                                                                                                                                                   
    def _reset_dofs(self, env_ids, dof_pos, dof_vel):
        
        # dof_pos_default = self.default_dof_pos + torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device) * self.default_dof_pos
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        # self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.5, (len(env_ids), self.num_dof), device=self.device)
        # self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def post_physics_step(self):
        # self._motion_sync()
        super().post_physics_step()

        # step motion lib
        self._motion_times += self._motion_dt
        
        # 安全地检查张量状态，避免在损坏的张量上操作
        try:
            # 检查张量形状是否匹配
            if self._motion_times.shape != self._motion_lengths.shape:
                print(f"[ERROR] Shape mismatch: _motion_times {self._motion_times.shape} vs _motion_lengths {self._motion_lengths.shape}")
                # 重新初始化motion_lengths
                self._motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
                print(f"[DEBUG] Reinitialized _motion_lengths with shape: {self._motion_lengths.shape}")
            
            # 检查NaN值
            if torch.any(torch.isnan(self._motion_times)) or torch.any(torch.isnan(self._motion_lengths)):
                print(f"[ERROR] NaN detected in motion tensors!")
                print(f"_motion_times NaN count: {torch.isnan(self._motion_times).sum()}")
                print(f"_motion_lengths NaN count: {torch.isnan(self._motion_lengths).sum()}")
                # 重新初始化motion_times
                self._motion_times = self._motion_lib.sample_time(self._motion_ids)
                print(f"[DEBUG] Reinitialized _motion_times with shape: {self._motion_times.shape}")
            
            # 执行motion时间更新
            self._motion_times[self._motion_times >= self._motion_lengths] = 0.
            
        except Exception as e:
            print(f"[ERROR] Failed to update motion times: {e}")
            # 完全重新初始化motion系统
            self._motion_times = self._motion_lib.sample_time(self._motion_ids)
            self._motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
            print(f"[DEBUG] Completely reinitialized motion system")
        self.update_demo_obs()
        # self.update_mimic_obs()
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.draw_rigid_bodies_demo()
            self.draw_rigid_bodies_actual()
            self._draw_goals()  # 添加绘制goals的调用

        return

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if self.common_step_counter % self.cfg.motion.resample_step_inplace_interval == 0:
            self.resample_step_inplace_ids()
    
    def resample_step_inplace_ids(self, ):
        self.step_inplace_ids = torch.rand(self.num_envs, device=self.device) < self.cfg.motion.step_inplace_prob
    
    def _randomize_gravity(self, external_force = None):
        if self.cfg.domain_rand.randomize_gravity and external_force is None:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity


        sim_params = self.gym.get_sim_params(self.sim)
        gravity = external_force + torch.Tensor([0, 0, -9.81]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.cfg.domain_rand.gravity_rand_interval = np.ceil(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)
        self.cfg.motion.resample_step_inplace_interval = np.ceil(self.cfg.motion.resample_step_inplace_interval_s / self.dt)

    def _update_goals(self):
        if self.use_ase_location_goals:
            # ===== ASE Location Goals更新逻辑 =====
            self._update_ase_location_goals()
            
            # 计算ASE goal的朝向差 (类似terrain项目的设计)
            ase_goal_rel = self._tar_pos - self.root_states[:, :2]
            norm = torch.norm(ase_goal_rel, dim=-1, keepdim=True)
            ase_goal_vec_norm = ase_goal_rel / (norm + 1e-5)
            self.ase_goal_yaw = torch.atan2(ase_goal_vec_norm[:, 1], ase_goal_vec_norm[:, 0])
            self.ase_delta_yaw = self.ase_goal_yaw - self.yaw
        else:
            # ===== 恢复原有的terrain goals更新逻辑 =====
            # 1. 首先调用父类方法更新terrain goals
            super()._update_goals()
        
        # 2. 然后更新motion demo goals
        # self.target_pos_abs = (self._curr_demo_root_pos - self.init_root_pos_global_demo + self.init_root_pos_global)[:, :2]
        # self.target_pos_rel = self.target_pos_abs - self.root_states[:, :2]
        reset_target_pos = self.episode_length_buf % (self.cfg.motion.global_keybody_reset_time // self.dt) == 0
        self.target_pos_abs[reset_target_pos] = self.root_states[reset_target_pos, :2]
        self.target_pos_abs += (self._curr_demo_root_vel * self.dt)[:, :2]
        self.target_pos_rel = global_to_local_xy(self.yaw[:, None], self.target_pos_abs - self.root_states[:, :2])
        # print(self.target_pos_rel[self.lookat_id])
        r, p, y = euler_from_quaternion(self._curr_demo_quat)
        self.target_yaw = y.clone()
        # self.desired_vel_scalar = torch.norm(self._curr_demo_obs_buf[:, self.num_dof:self.num_dof+2], dim=-1)

        # 维护“上次达成目标时的步数”：如发现cur_goal_idx增长则更新
        try:
            if not hasattr(self, '_last_goal_idx'):
                self._last_goal_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            if not hasattr(self, '_last_goal_step'):
                self._last_goal_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            progressed = self.cur_goal_idx > self._last_goal_idx
            if torch.any(progressed):
                self._last_goal_idx[progressed] = self.cur_goal_idx[progressed]
                self._last_goal_step[progressed] = self.episode_length_buf[progressed]
        except Exception:
            pass

    def _update_ase_location_goals(self):
        """更新ASE风格的location goals"""
        # 检查是否需要更新目标
        reset_task_mask = self.episode_length_buf >= self._tar_change_steps
        reset_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        
        if len(reset_env_ids) > 0:
            self._reset_ase_location_goals(reset_env_ids)
        
        # 检查是否到达目标
        current_pos = self.root_states[:, :2]
        distance_to_goal = torch.norm(current_pos - self._tar_pos, dim=1)
        reached_goal = distance_to_goal < self._goal_reach_threshold
        
        # 更新目标到达计时器
        self._goal_reach_timer[reached_goal] += self.dt
        self._goal_reach_timer[~reached_goal] = 0.0
        
        # 如果到达目标超过一定时间，生成新目标
        goal_reach_time_threshold = 1.0  # 到达目标后1秒生成新目标
        should_generate_new_goal = (reached_goal & (self._goal_reach_timer > goal_reach_time_threshold))
        new_goal_env_ids = should_generate_new_goal.nonzero(as_tuple=False).flatten()
        
        if len(new_goal_env_ids) > 0:
            self._reset_ase_location_goals(new_goal_env_ids)
    
    def update_demo_obs(self):
        # 禁用模仿学习观测更新，避免CUDA错误
        # 原始代码已注释，因为用户不需要模仿学习观测
        # demo_motion_times = self._motion_demo_offsets + self._motion_times[:, None]  # [num_envs, demo_dim]
        # root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, local_key_body_pos \
        #     = self._motion_lib.get_motion_state(self._motion_ids.repeat_interleave(self._motion_num_future_steps), demo_motion_times.flatten(), get_lbp=True)
        # dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)
        # 
        # self._curr_demo_root_pos[:] = root_pos.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]
        # self._curr_demo_quat[:] = root_rot.view(self.num_envs, self._motion_num_future_steps, 4)[:, 0, :]
        # self._curr_demo_root_vel[:] = root_vel.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]
        # self._curr_demo_keybody[:] = local_key_body_pos[:, self._key_body_ids_sim_subset].view(self.num_envs, self._motion_num_future_steps, self._num_key_bodies, 3)[:, 0, :, :]
        # self._in_place_flag = (torch.norm(self._curr_demo_root_vel, dim=-1) < 0.2)
        # # for i in range(13):
        # #     feet_pos_global = key_pos[:, i]# - root_pos + self.root_states[:, :3]
        # #     pose = gymapi.Transform(gymapi.Vec3(feet_pos_global[self.lookat_id, 0], feet_pos_global[self.lookat_id, 1], feet_pos_global[self.lookat_id, 2]), r=None)
        # #     gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        # demo_obs = build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos[:, self._dof_ids_subset], dof_vel, key_pos, local_key_body_pos[:, self._key_body_ids_sim_subset, :], self._dof_offsets)
        # self._demo_obs_buf[:] = demo_obs.view(self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo)[:]
        
        # 保持demo观测为零值，不进行任何更新
        pass
    
    def compute_obs_buf(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        
        if self.use_ase_location_goals:
            # 完全按照terrain项目的结构
            goal_obs1 = 0 * self.ase_delta_yaw[:, None]  # [1] 置零 (对应terrain的0*delta_yaw)
            goal_obs2 = self.ase_delta_yaw[:, None]      # [1] ASE goal朝向差 (对应terrain的delta_yaw)
            goal_obs3 = self.ase_delta_yaw[:, None]      # [1] 重复使用 (对应terrain的delta_next_yaw)
        else:
            # ===== Terrain Goals观测构建 (兼容ASE goals的观测结构) =====
            # 计算terrain goals的朝向差 (类似terrain项目的实现)
            if hasattr(self, 'cur_goals') and self.cur_goals is not None:
                # 当前目标朝向差
                cur_goal_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
                norm = torch.norm(cur_goal_rel, dim=-1, keepdim=True)
                cur_goal_vec_norm = cur_goal_rel / (norm + 1e-5)
                cur_goal_yaw = torch.atan2(cur_goal_vec_norm[:, 1], cur_goal_vec_norm[:, 0])
                terrain_delta_yaw = cur_goal_yaw - self.yaw
                
                # 下一个目标朝向差 (如果有的话)
                if hasattr(self, 'next_goals') and self.next_goals is not None:
                    next_goal_rel = self.next_goals[:, :2] - self.root_states[:, :2]
                    norm = torch.norm(next_goal_rel, dim=-1, keepdim=True)
                    next_goal_vec_norm = next_goal_rel / (norm + 1e-5)
                    next_goal_yaw = torch.atan2(next_goal_vec_norm[:, 1], next_goal_vec_norm[:, 0])
                    terrain_delta_next_yaw = next_goal_yaw - self.yaw
                else:
                    terrain_delta_next_yaw = terrain_delta_yaw  # 如果没有下一个目标，使用当前目标
            else:
                # 如果没有terrain goals，使用零值
                terrain_delta_yaw = torch.zeros_like(self.yaw)
                terrain_delta_next_yaw = torch.zeros_like(self.yaw)
            
            # 构建观测 (保持与ASE goals相同的结构)
            goal_obs1 = 0 * terrain_delta_yaw[:, None]        # [1] 置零
            goal_obs2 = terrain_delta_yaw[:, None]            # [1] 当前目标朝向差
            goal_obs3 = terrain_delta_next_yaw[:, None]       # [1] 下一个目标朝向差
            
        return torch.cat((#motion_id_one_hot,
                            self.base_ang_vel  * self.obs_scales.ang_vel,   #[3] 角速度
                            imu_obs,    #[2] roll, pitch
                            goal_obs1,  #[1] 置零的朝向差
                            goal_obs2,  #[1] 当前目标朝向差
                            goal_obs3,  #[1] 下一个目标朝向差
                            # 添加命令相关观测（3维）
                            0*self.commands[:, 0:2],  #[2] 置零的命令
                            self.commands[:, 0:1],    #[1] 线速度命令
                            # 添加环境类别观测（2维）
                            (self.env_class != 17).float()[:, None],  #[1] 环境类别标志1
                            (self.env_class == 17).float()[:, None],  #[1] 环境类别标志2
                            # self.target_pos_rel,  
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),  #[19] DOF位置
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),                              #[19] DOF速度
                            self.reindex(self.action_history_buf[:, -1]),                                     #[19] 上一动作
                            self.reindex_feet(self.contact_filt.float()*0-0.5),                               #[2] 接触状态
                            ),dim=-1)
    
    def compute_obs_demo(self):
        # 返回零填充的观测，保持维度一致，不影响AMP功能
        return torch.zeros((self.num_envs, self.cfg.env.n_demo), device=self.device)
    
    def compute_observations(self):
        # motion_id_one_hot = torch.zeros((self.num_envs, self._motion_lib.num_motions()), device=self.device)
        # motion_id_one_hot[torch.arange(self.num_envs, device=self.device), self._motion_ids] = 1.
        
        obs_buf = self.compute_obs_buf()

        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_scale
        
        obs_demo = self.compute_obs_demo()
        
        # obs_demo[:, :] = 0
        # obs_demo[:, -3*len(self._key_body_ids_sim_subset)-1] = 1
        # obs_demo[:, self._n_demo_dof:self._n_demo_dof+8] = 1 #self.rand_vx_cmd
        # obs_demo[:, -3*len(self._key_body_ids_sim_subset):] = torch.tensor([ 0.0049,  0.1554,  0.4300,  
        #                                                                      0.0258,  0.2329,  0.1076,  
        #                                                                      0.3195,  0.2040,  0.0537,  
        #                                                                      0.0061, -0.1553,  0.4300,  
        #                                                                      0.0292, -0.2305,  0.1076,  
        #                                                                      0.3225, -0.1892,  0.0598], device=self.device)
        motion_features = self.obs_history_buf[:, -self.cfg.env.prop_hist_len:].flatten(start_dim=1)#self._demo_obs_buf[:, 2:, :].clone().flatten(start_dim=1) 
        priv_explicit = torch.cat((0*self.base_lin_vel * self.obs_scales.lin_vel,
                                #    global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3], self.root_states[:, :3]).view(self.num_envs, -1),
                                  ), dim=-1)
        # 简化的调试信息：只检查基本属性，避免触发CUDA错误
        if hasattr(self, 'motor_strength'):
            try:
                print(f"[DEBUG] motor_strength shape: {self.motor_strength.shape}")
                print(f"[DEBUG] motor_strength device: {self.motor_strength.device}")
            except Exception as e:
                print(f"[ERROR] Failed to access motor_strength basic info: {e}")
        else:
            print(f"[ERROR] motor_strength not found!")
        
        # 安全地使用motor_strength
        try:
            # 先尝试简单的操作来检查CUDA内核状态
            _ = self.motor_strength[0].shape
            _ = self.motor_strength[1].shape
            
            priv_latent = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1, 
                self.motor_strength[1] - 1
            ), dim=-1)
        except Exception as e:
            print(f"[ERROR] Failed to use motor_strength in priv_latent: {e}")
            print(f"[ERROR] CUDA kernel appears to be corrupted, using fallback values")
            
            # 使用默认值作为fallback，避免访问损坏的张量
            fallback_motor_0 = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
            fallback_motor_1 = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
            
            priv_latent = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                fallback_motor_0, 
                fallback_motor_1
            ), dim=-1)
        
        # 获取 scan 观测（地形高度）
        scan_obs = None
        if self.cfg.terrain.measure_heights:
            # 调用基类方法获取地形高度
            self.measured_heights = self._get_heights()
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            scan_obs = heights
            self.obs_buf = torch.cat([motion_features, obs_buf, obs_demo, scan_obs, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            # 如果没有地形高度测量，创建零填充的 scan 观测
            # 安全检查：确保 n_scan 属性存在
            n_scan = getattr(self.cfg, 'n_scan', 132)  # 默认值132，如果配置中没有定义
            scan_obs = torch.zeros((self.num_envs, n_scan), device=self.device)
            self.obs_buf = torch.cat([motion_features, obs_buf, obs_demo, scan_obs, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        # 添加边界检查，防止CUDA索引越界
        contact_filt_safe = self.contact_filt.float()
        # 确保contact_filt的维度正确
        if contact_filt_safe.dim() != 2 or contact_filt_safe.shape[0] != self.num_envs:
            print(f"Warning: contact_filt shape mismatch: {contact_filt_safe.shape}, expected: ({self.num_envs}, {contact_filt_safe.shape[1] if contact_filt_safe.dim() > 1 else 'unknown'})")
            # 如果维度不匹配，创建一个安全的张量
            contact_filt_safe = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        
        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([contact_filt_safe] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                contact_filt_safe.unsqueeze(1)
            ], dim=1)
        )

    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids
        # print(self._motion_times[self.lookat_id])
        # motion_times = self.episode_length_buf * self._motion_dt

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
           = self._motion_lib.get_motion_state(motion_ids, self._motion_times)
        
        root_pos[:, :2] = (self._curr_demo_root_pos - self.init_root_pos_global_demo + self.init_root_pos_global)[:, :2]
        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:10] = root_vel
        self.root_states[env_ids, 10:13] = root_ang_vel

        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel
        return

    def check_termination(self):
        """ Check if environments need to be reset
        """
        print(f"[DEBUG] check_termination called, episode_length_buf: {self.episode_length_buf[0].item()}")
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # roll_cutoff = torch.abs(self.roll) > 1.0
        # pitch_cutoff = torch.abs(self.pitch) > 1.0
        # height_cutoff = self.root_states[:, 2] < 0.5

        # Optionally disable mimic deviation termination
        if getattr(self.cfg.env, 'enable_mimic_dev_termination', True):
            dof_dev = self._reward_tracking_demo_dof_pos() < 0.1
            self.reset_buf |= dof_dev
        
        # 调试：输出关键状态信息
        self.extras['debug_cur_goals_exists'] = hasattr(self, 'cur_goals') and self.cur_goals is not None
        self.extras['debug_cur_goal_idx'] = getattr(self, 'cur_goal_idx', torch.tensor(-1, device=self.device))
        self.extras['debug_episode_length'] = self.episode_length_buf
        if hasattr(self, '_last_goal_step'):
            self.extras['debug_last_goal_step'] = self._last_goal_step
            self.extras['debug_elapsed_steps'] = self.episode_length_buf - self._last_goal_step

        # 早停机制：如果机器人跑到当前目标前面太多，直接终止
        if hasattr(self, 'cur_goals') and self.cur_goals is not None:
            # 计算机器人当前位置到当前目标的方向向量
            robot_to_goal = self.cur_goals[:, :2] - self.root_states[:, :2]
            # 计算机器人当前速度方向
            robot_velocity = self.base_lin_vel[:, :2]
            
            # 如果机器人已经跑过了目标（在目标后面），且还在向前跑，则终止
            # 使用点积判断：如果速度方向与目标方向相反，说明跑过了
            dot_product = torch.sum(robot_to_goal * robot_velocity, dim=1)
            distance_to_goal = torch.norm(robot_to_goal, dim=1)
            
            # 使用配置参数控制早停条件
            early_stop = (distance_to_goal < self.cfg.env.early_stop_distance_threshold) & \
                        (dot_product < self.cfg.env.early_stop_velocity_threshold)
            self.reset_buf |= early_stop

            # 未踩到下一个点直接早停：依据cur_goal_idx在过去no_next_goal_time_s是否递增
            try:
                window_s = float(getattr(self.cfg.env, 'no_next_goal_time_s', 8.0))
                window_steps = int(window_s / self.dt)
                if hasattr(self, '_last_goal_step') and hasattr(self, '_last_goal_idx') and window_steps > 0:
                    not_finished = self.cur_goal_idx < self.cfg.terrain.num_goals
                    elapsed = (self.episode_length_buf - self._last_goal_step) >= window_steps
                    no_progress = self.cur_goal_idx == self._last_goal_idx
                    no_next_goal_window = not_finished & elapsed & no_progress
                    self.reset_buf |= no_next_goal_window
                    # 调试输出
                    self.extras['no_next_goal_window_steps'] = torch.tensor(window_steps, device=self.device)
                    self.extras['no_next_goal_elapsed'] = (self.episode_length_buf - self._last_goal_step)
                    self.extras['no_next_goal_trigger'] = no_next_goal_window
            except Exception:
                pass

        # demo_dofs = self._curr_demo_obs_buf[:, :self.num_dof]
        # ref_deviation = torch.norm(self.dof_pos - demo_dofs, dim=1) >= self.dof_term_threshold
        # self.reset_buf |= ref_deviation
        
        # height_dev = torch.abs(self.root_states[:, 2] - self._curr_demo_root_pos[:, 2]) >= self.height_term_threshold
        # self.reset_buf |= height_dev

        # yaw_dev = self._reward_tracking_demo_yaw() < self.yaw_term_threshold
        # self.reset_buf |= yaw_dev

        # ref_keybody_dev = self._reward_tracking_demo_key_body() < 0.2
        # self.reset_buf |= ref_keybody_dev

        # ref_deviation = (torch.norm(self.dof_pos - demo_dofs, dim=1) >= 1.5) & \
        #                 (self._motion_difficulty < 3)
        # self.reset_buf |= ref_deviation
        
        # ref_keybody_dev = (self._reward_tracking_demo_key_body() < 0.3) & \
        #                   (self._motion_difficulty < 3)
        # self.reset_buf |= ref_keybody_dev

        # Early stop if forward displacement over 8s is too small
        try:
            window_steps = int(8.0 / self.dt)
            disp_thresh = getattr(self.cfg.env, 'min_forward_disp_8s', None)
            if disp_thresh is not None and window_steps > 1:
                # approximate forward displacement using base x-position change
                # require that we have buffer of previous root positions; if not, build a simple ring buffer
                if not hasattr(self, '_prev_root_x_hist') or self._prev_root_x_hist.shape[1] != window_steps:
                    self._prev_root_x_hist = torch.zeros(self.num_envs, window_steps, device=self.device)
                    self._prev_hist_idx = 0
                # push current x
                self._prev_root_x_hist[:, self._prev_hist_idx] = self.root_states[:, 0]
                self._prev_hist_idx = (self._prev_hist_idx + 1) % window_steps
                # compute displacement using current minus value from window_steps ago
                idx_past = self._prev_hist_idx  # this is the position to be overwritten next, which holds x(t-8s)
                x_past = self._prev_root_x_hist[:, idx_past]
                x_now = self.root_states[:, 0]
                forward_disp = x_now - x_past
                small_disp = forward_disp < disp_thresh
                # avoid triggering during warmup before buffer filled
                warmup = self.episode_length_buf < window_steps
                small_disp = small_disp & (~warmup)
                self.reset_buf |= small_disp
        except Exception:
            pass

        motion_end = self.episode_length_buf * self.dt >= self._motion_lengths
        self.reset_buf |= motion_end

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf |= motion_end

        self.reset_buf |= self.time_out_buf
        # self.reset_buf |= roll_cutoff
        # self.reset_buf |= pitch_cutoff
        # self.reset_buf |= height_cutoff

    ######### demonstrations #########
    # def get_demo_obs(self, ):
    #     demo_motion_times = self._motion_demo_offsets + self._motion_times[:, None]  # [num_envs, demo_dim]
    #     # get the motion state at the demo times
    #     root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
    #         = self._motion_lib.get_motion_state(self._motion_ids.repeat(self._motion_num_future_steps), demo_motion_times.flatten())
    #     dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)
        
    #     demo_obs = build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, self._dof_offsets)
    #     return demo_obs
    
    # def get_curr_demo(self):
    #     root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
    #         = self._motion_lib.get_motion_state(self._motion_ids, self._motion_times)
    #     dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)
    #     demo_obs = build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, self._dof_offsets)
    #     return demo_obs
    
    
    ######### utils #########
    
    def reindex_dof_pos_vel(self, dof_pos, dof_vel):
        dof_pos = reindex_motion_dof(dof_pos, self.dof_indices_sim, self.dof_indices_motion, self._valid_dof_body_ids)
        dof_vel = reindex_motion_dof(dof_vel, self.dof_indices_sim, self.dof_indices_motion, self._valid_dof_body_ids)
        return dof_pos, dof_vel

    def draw_rigid_bodies_demo(self, ):
        geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(0, 1, 0))
        local_body_pos = self._curr_demo_keybody.clone().view(self.num_envs, self._num_key_bodies, 3)
        if self.cfg.motion.global_keybody:
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)
        else:
            curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)
        global_body_pos = local_to_global(self._curr_demo_quat, local_body_pos, curr_demo_xyz)
        for i in range(global_body_pos.shape[1]):
            pose = gymapi.Transform(gymapi.Vec3(global_body_pos[self.lookat_id, i, 0], global_body_pos[self.lookat_id, i, 1], global_body_pos[self.lookat_id, i, 2]), r=None)
            gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def draw_rigid_bodies_actual(self, ):
        geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(1, 0, 0))
        rigid_body_pos = self.rigid_body_states[:, self._key_body_ids_sim, :3].clone()
        for i in range(rigid_body_pos.shape[1]):
            pose = gymapi.Transform(gymapi.Vec3(rigid_body_pos[self.lookat_id, i, 0], rigid_body_pos[self.lookat_id, i, 1], rigid_body_pos[self.lookat_id, i, 2]), r=None)
            gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def _draw_goals(self):
        """绘制goals"""
        if self.use_ase_location_goals:
            # ===== ASE Location Goals可视化 =====
            self._draw_ase_location_goals()
        else:
            # ===== 恢复原有的terrain goals可视化 =====
            # 直接调用父类方法
            super()._draw_goals()

    def _draw_ase_location_goals(self):
        """绘制ASE风格的location goals"""
        if self.viewer is None:
            return
            
        # 绘制目标点
        goal_pos = self._tar_pos[self.lookat_id].cpu().numpy()
        goal_z = 0.0  # 在平地上
        
        # 绘制目标点（红色球体）
        sphere_geom = gymutil.WireframeSphereGeometry(0.2, 16, 16, None, color=(1, 0, 0))
        pose = gymapi.Transform(gymapi.Vec3(goal_pos[0], goal_pos[1], goal_z), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
        # 暂时不绘制连线，避免add_lines错误
        # TODO: 后续可以添加连线功能
    
    ######### Rewards #########
    def compute_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew #if "demo" not in name else 0  # log demo rew but do not include in additative reward
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if self.cfg.rewards.clip_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=-0.5)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        
    def _reward_tracking_demo_goal_vel(self):
        norm = torch.norm(self._curr_demo_root_vel[:, :3], dim=-1, keepdim=True)
        target_vec_norm = self._curr_demo_root_vel[:, :3] / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:10]
        norm_squeeze = norm.squeeze(-1)
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), norm_squeeze) / (norm_squeeze + 1e-5)

        rew_zeros = torch.exp(-4*torch.norm(cur_vel, dim=-1))
        small_cmd_ids = (norm<0.1).squeeze(-1)
        rew[small_cmd_ids] = rew_zeros[small_cmd_ids]
        # return torch.exp(-2 * torch.norm(cur_vel - self._curr_demo_root_vel[:, :2], dim=-1))
        return rew.squeeze(-1)

    def _reward_tracking_vx(self):
        rew = torch.minimum(self.base_lin_vel[:, 0], self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        # print("vx rew", rew, self.base_lin_vel[:, 0], self.commands[:, 0])
        return rew
    
    def _reward_tracking_ang_vel(self):
        rew = torch.minimum(self.base_ang_vel[:, 2], self.commands[:, 2]) / (self.commands[:, 2] + 1e-5)
        return rew
    
    def _reward_tracking_demo_yaw(self):
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        # print("yaw rew", rew, self.target_yaw, self.yaw)
        return rew

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        # print("lower dof pos error: ", self.dof_pos - self.dof_pos_limits[:, 0])
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        # print("upper dof pos error: ", self.dof_pos - self.dof_pos_limits[:, 1])
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_demo_dof_pos(self):
        demo_dofs = self._curr_demo_obs_buf[:, :self._n_demo_dof]
        dof_pos = self.dof_pos[:, self._dof_ids_subset]
        rew = torch.exp(-0.7 * torch.norm((dof_pos - demo_dofs), dim=1))
        # print(rew[self.lookat_id].cpu().numpy())
        # print("dof_pos", dof_pos)
        # print("demo_dofs", demo_dofs)
        return rew

    def _reward_tracking_demo_lower_body_dof_pos(self):
        """
        下半身DOF跟踪奖励函数
        只跟踪腿部DOF (索引0-9) 和躯干DOF (索引10)
        总共11个DOF: 左腿5个 + 右腿5个 + 躯干1个
        
        由于当前演示观察中只包含躯干和手臂DOF，我们需要从原始运动数据中获取腿部DOF
        """
        # 下半身DOF索引: 左腿(0-4) + 右腿(5-9) + 躯干(10)
        lower_body_dof_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], device=self.device)
        
        # 获取当前机器人的下半身DOF位置
        current_lower_body_dofs = self.dof_pos[:, lower_body_dof_indices]
        
        # 从演示观察中获取躯干DOF (索引10)
        demo_dofs = self._curr_demo_obs_buf[:, :self._n_demo_dof]  # 当前演示DOF
        # 演示DOF中第一个是躯干DOF (对应_dof_ids_subset中的索引10)
        demo_torso_dof = demo_dofs[:, 0:1]  # 躯干DOF
        
        # 获取原始运动数据中的腿部DOF
        # 我们需要从运动库中获取完整的DOF数据
        demo_motion_times = self._motion_demo_offsets + self._motion_times[:, None]
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, local_key_body_pos \
            = self._motion_lib.get_motion_state(self._motion_ids.repeat_interleave(self._motion_num_future_steps), demo_motion_times.flatten(), get_lbp=True)
        
        # 重新索引DOF以匹配机器人结构
        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)
        
        # 提取腿部DOF (索引0-9)
        demo_leg_dofs = dof_pos.view(self.num_envs, self._motion_num_future_steps, -1)[:, 0, :10]  # 取第一个时间步的腿部DOF
        
        # 组合演示的下半身DOF
        demo_lower_body_dofs = torch.cat([demo_leg_dofs, demo_torso_dof], dim=1)
        
        # 计算下半身DOF位置误差
        dof_error = torch.norm((current_lower_body_dofs - demo_lower_body_dofs), dim=1)
        
        # 使用指数函数计算奖励，权重可以调整
        rew = torch.exp(-0.7 * dof_error)
        
        return rew

    # def _reward_tracking_demo_dof_vel(self):
    #     demo_dof_vel = self._curr_demo_obs_buf[:, self.num_dof:self.num_dof*2]
    #     rew = torch.exp(- 0.01 * torch.norm(self.dof_vel - demo_dof_vel, dim=1))
    #     return rew
    
    def _reward_stand_still(self):
        dof_pos_error = torch.norm((self.dof_pos - self.default_dof_pos)[:, :11], dim=1)
        dof_vel_error = torch.norm(self.dof_vel[:, :11], dim=1)
        rew = torch.exp(- 0.1*dof_vel_error) * torch.exp(- dof_pos_error) 
        rew[~self._in_place_flag] = 0
        return rew
    
    def _reward_tracking_lin_vel(self):
        demo_vel = self._curr_demo_obs_buf[:, self._n_demo_dof:self._n_demo_dof+3]
        demo_vel[self._in_place_flag] = 0
        rew = torch.exp(- 4 * torch.norm(self.base_lin_vel - demo_vel, dim=1))
        return rew

    def _reward_tracking_demo_ang_vel(self):
        demo_ang_vel = self._curr_demo_obs_buf[:, self._n_demo_dof+3:self._n_demo_dof+6]
        rew = torch.exp(-torch.norm(self.base_ang_vel - demo_ang_vel, dim=1))
        return rew

    def _reward_tracking_demo_roll_pitch(self):
        demo_roll_pitch = self._curr_demo_obs_buf[:, self._n_demo_dof+6:self._n_demo_dof+8]
        cur_roll_pitch = torch.stack((self.roll, self.pitch), dim=1)
        rew = torch.exp(-torch.norm(cur_roll_pitch - demo_roll_pitch, dim=1))
        return rew
    
    def _reward_tracking_demo_height(self):
        demo_height = self._curr_demo_obs_buf[:, self._n_demo_dof+8]
        cur_height = self.root_states[:, 2]
        rew = torch.exp(- 4 * torch.abs(cur_height - demo_height))
        return rew
    
    def _reward_tracking_demo_key_body(self):
        # demo_key_body_pos_local = self._curr_demo_obs_buf[:, self.num_dof*2+8:].view(self.num_envs, self._num_key_bodies, 3)[:,self._key_body_ids_sim_subset,:].view(self.num_envs, -1)
        # cur_key_body_pos_local = global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3], self.root_states[:, :3]).view(self.num_envs, -1)
        
        demo_key_body_pos_local = self._curr_demo_keybody.view(self.num_envs, self._num_key_bodies, 3)
        if self.cfg.motion.global_keybody:
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)
        else:
            curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)
        demo_global_body_pos = local_to_global(self._curr_demo_quat, demo_key_body_pos_local, curr_demo_xyz).view(self.num_envs, -1)
        cur_global_body_pos = self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3].view(self.num_envs, -1)

        # cur_local_body_pos = global_to_local(self.base_quat, cur_global_body_pos.view(self.num_envs, -1, 3), self.root_states[:, :3]).view(self.num_envs, -1)
        # print(cur_local_body_pos)
        rew = torch.exp(-torch.norm(cur_global_body_pos - demo_global_body_pos, dim=1))
        # print("key body rew", rew[self.lookat_id].cpu().numpy())
        return rew

    def _reward_tracking_mul(self):
        rew_key_body = self._reward_tracking_demo_key_body()
        rew_roll_pitch = self._reward_tracking_demo_roll_pitch()
        rew_ang_vel = self._reward_tracking_demo_yaw()
        # rew_dof_vel = self._reward_tracking_demo_dof_vel()
        rew_dof_pos = self._reward_tracking_demo_dof_pos()
        # rew_goal_vel = self._reward_tracking_lin_vel()#self._reward_tracking_demo_goal_vel()
        rew = rew_key_body * rew_roll_pitch * rew_ang_vel * rew_dof_pos# * rew_dof_vel
        # print(self._curr_demo_obs_buf[:, self.num_dof:self.num_dof+3][self.lookat_id], self.base_lin_vel[self.lookat_id])
        return rew
    # def _reward_tracking_demo_vel(self):
    #     demo_vel = self.get_curr_demo()[:, self.num_dof:]
    def _reward_feet_drag(self):
        # print(contact_bool)
        # contact_forces = self.contact_forces[:, self.feet_indices, 2]
        # print(contact_forces[self.lookat_id], self.force_sensor_tensor[self.lookat_id, :, 2])
        # print(self.contact_filt[self.lookat_id])
        feet_xyz_vel = torch.abs(self.rigid_body_states[:, self.feet_indices, 7:10]).sum(dim=-1)
        dragging_vel = self.contact_filt * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        # print(rew[self.lookat_id].cpu().numpy(), self.contact_filt[self.lookat_id].cpu().numpy(), feet_xy_vel[self.lookat_id].cpu().numpy())
        return rew
    
    def _reward_energy(self):
        return torch.norm(torch.abs(self.torques * self.dof_vel), dim=-1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        rew_airTime[self._in_place_flag] = 0
        return rew_airTime

    def _reward_feet_height(self):
        feet_height = self.rigid_body_states[:, self.feet_indices, 2]
        rew = torch.clamp(torch.norm(feet_height, dim=-1) - 0.2, max=0)
        rew[self._in_place_flag] = 0
        # print("height: ", rew[self.lookat_id])
        return rew
    
    def _reward_feet_force(self):
        rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
        rew[rew < 500] = 0
        rew[rew > 500] -= 500
        rew[self._in_place_flag] = 0
        # print(rew[self.lookat_id])
        # print(self.dof_names)
        return rew

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, :11], dim=1)
        return dof_error
    
    def _reward_next_goal_direction(self):
        """奖励机器人朝向下一个目标的方向"""
        if self.use_ase_location_goals:
            # ===== ASE Location Goals方向奖励 (使用点积方法) =====
            # 计算目标方向向量
            tar_dir = self._tar_pos - self.root_states[:, :2]
            tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
            
            # 计算机器人当前朝向 (前方方向)
            heading_rot = self.base_quat  # 机器人朝向四元数
            facing_dir = torch.zeros_like(self.root_states[:, :3])
            facing_dir[:, 0] = 1.0  # 机器人前方方向 (x轴)
            facing_dir = quat_apply(heading_rot, facing_dir)
            
            # 计算朝向相似度 (点积)
            facing_err = torch.sum(tar_dir * facing_dir[:, :2], dim=-1)
            
            # 奖励：只奖励正向朝向，负向朝向奖励为0
            rew = torch.clamp_min(facing_err, 0.0)
            return rew
        else:
            # ===== 恢复原有的terrain goals方向奖励 =====
            # 获取下一个目标的相对位置
            next_goal_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
            
            # 计算机器人当前朝向 - 修复batch维度问题
            forward_vec = quat_apply(self.base_quat, torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, 3))
            
            # 计算朝向下一个目标的角度
            goal_angle = torch.atan2(next_goal_rel[:, 1], next_goal_rel[:, 0])
            robot_angle = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])
            
            # 角度差（考虑周期性）
            angle_diff = torch.abs(torch.atan2(torch.sin(goal_angle - robot_angle), torch.cos(goal_angle - robot_angle)))
            
            # 奖励：角度差越小，奖励越高
            rew = torch.exp(-2.0 * angle_diff)
            return rew

    def _reward_next_goal_distance(self):
        """奖励机器人接近下一个目标"""
        if self.use_ase_location_goals:
            # ===== ASE Location Goals距离奖励 =====
            # 计算到目标的距离
            distance_to_goal = torch.norm(self._tar_pos - self.root_states[:, :2], dim=1)
            
            # 奖励：距离越近，奖励越高
            rew = torch.exp(-0.5 * distance_to_goal)
            return rew
        else:
            # ===== 恢复原有的terrain goals距离奖励 =====
            # 计算到下一个目标的距离
            distance_to_next = torch.norm(self.cur_goals[:, :2] - self.root_states[:, :2], dim=1)
            
            # 奖励：距离越近，奖励越高
            rew = torch.exp(-0.5 * distance_to_next)
            return rew

    def _reward_next_goal_velocity(self):
        """奖励机器人以合适速度朝向下一个目标移动"""
        if self.use_ase_location_goals:
            # ===== ASE Location Goals速度奖励 =====
            # 计算目标方向
            tar_dir = self._tar_pos - self.root_states[:, :2]
            tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
            
            # 获取机器人当前速度
            root_vel = self.root_states[:, 7:10]
            
            # 计算朝向目标方向的速度分量
            tar_dir_speed = torch.sum(tar_dir * root_vel[:, :2], dim=-1)
            
            # 使用配置的目标速度
            tar_speed = self._tar_speed
            
            # 速度误差（只惩罚速度过慢，不惩罚过快）
            tar_vel_err = tar_speed - tar_dir_speed
            tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
            
            # 速度奖励：指数衰减
            vel_reward = torch.exp(-4.0 * (tar_vel_err * tar_vel_err))
            
            # 如果朝向错误方向（速度为负），奖励为0
            speed_mask = tar_dir_speed <= 0
            vel_reward[speed_mask] = 0
            
            # 距离阈值内给满分（接近目标时不需要高速）
            distance_to_goal = torch.norm(self._tar_pos - self.root_states[:, :2], dim=1)
            dist_mask = distance_to_goal < self._goal_reach_threshold
            vel_reward[dist_mask] = 1.0
            
            return vel_reward
        else:
            # ===== 恢复原有的terrain goals速度奖励 =====
            # 计算目标方向
            tar_dir = self.cur_goals[:, :2] - self.root_states[:, :2]
            tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
            
            # 获取机器人当前速度
            root_vel = self.root_states[:, 7:10]
            
            # 计算朝向目标方向的速度分量
            tar_dir_speed = torch.sum(tar_dir * root_vel[:, :2], dim=-1)
            
            # 目标速度（可配置）
            tar_speed = 1.0  # 1.0 m/s
            
            # 速度误差（只惩罚速度过慢，不惩罚过快）
            tar_vel_err = tar_speed - tar_dir_speed
            tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
            
            # 速度奖励：指数衰减
            vel_reward = torch.exp(-4.0 * (tar_vel_err * tar_vel_err))
            
            # 如果朝向错误方向（速度为负），奖励为0
            speed_mask = tar_dir_speed <= 0
            vel_reward[speed_mask] = 0
            
            # 距离阈值内给满分（接近目标时不需要高速）
            distance_to_next = torch.norm(self.cur_goals[:, :2] - self.root_states[:, :2], dim=1)
            dist_mask = distance_to_next < 0.5  # 0.5米阈值
            vel_reward[dist_mask] = 1.0
            
            return vel_reward

    def _reward_reverse_velocity_penalty(self):
        """惩罚反向移动和静止状态"""
        if self.use_ase_location_goals:
            # ===== ASE Location Goals反向速度惩罚 =====
            # 计算目标方向
            tar_dir = self._tar_pos - self.root_states[:, :2]
            tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
            
            # 获取机器人当前速度
            root_vel = self.root_states[:, 7:10]
            
            # 计算朝向目标方向的速度分量
            tar_dir_speed = torch.sum(tar_dir * root_vel[:, :2], dim=-1)
            
            # 计算总速度大小
            total_speed = torch.norm(root_vel[:, :2], dim=-1)
            
            # 反向速度惩罚：速度为负时给予惩罚
            reverse_penalty = torch.clamp_min(-tar_dir_speed, 0.0)
            
            # 静止状态惩罚：速度过小时给予惩罚
            still_penalty = torch.clamp_min(0.1 - total_speed, 0.0)
            
            # 组合惩罚
            penalty = reverse_penalty + still_penalty
            
            return penalty
        else:
            # ===== 恢复原有的terrain goals反向速度惩罚 =====
            # 计算目标方向
            tar_dir = self.cur_goals[:, :2] - self.root_states[:, :2]
            tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
            
            # 获取机器人当前速度
            root_vel = self.root_states[:, 7:10]
            
            # 计算朝向目标方向的速度分量
            tar_dir_speed = torch.sum(tar_dir * root_vel[:, :2], dim=-1)
            
            # 计算总速度大小
            total_speed = torch.norm(root_vel[:, :2], dim=-1)
            
            # 反向速度惩罚：速度为负时给予惩罚
            reverse_penalty = torch.clamp_min(-tar_dir_speed, 0.0)
            
            # 静止状态惩罚：速度过小时给予惩罚
            still_penalty = torch.clamp_min(0.1 - total_speed, 0.0)
            
            # 组合惩罚
            penalty = reverse_penalty + still_penalty
            
            return penalty

    def _reward_feet_parkour_penalty(self):
        """惩罚脚部接触非parkour区域（仅在第一阶段训练时启用）"""
        # 检查是否启用两阶段训练且当前为第一阶段
        if not (hasattr(self.cfg.terrain, 'two_stage_training') and 
                self.cfg.terrain.two_stage_training and 
                self.cfg.terrain.training_stage == 1):
            return torch.zeros(self.num_envs, device=self.device)
        
        # 检查地形是否有有效站立掩码
        if not hasattr(self.terrain, 'valid_standing_mask'):
            return torch.zeros(self.num_envs, device=self.device)
        
        # 获取机器人位置
        robot_pos = self.root_states[:, :2]  # [num_envs, 2]
        
        # 检查机器人是否在parkour地形区域内（从起始平台开始）
        # parkour地形通常从x=20开始（起始平台）
        parkour_start_x = 20.0  # 起始平台位置（米）
        parkour_start_y = -2.0  # parkour地形宽度的一半
        parkour_end_y = 2.0
        
        # 只对在parkour区域内的机器人计算惩罚
        in_parkour_area = ((robot_pos[:, 0] >= parkour_start_x) & 
                          (robot_pos[:, 1] >= parkour_start_y) & 
                          (robot_pos[:, 1] <= parkour_end_y))
        
        if not torch.any(in_parkour_area):
            return torch.zeros(self.num_envs, device=self.device)
        
        # 获取脚部位置
        feet_pos = self.rigid_body_states[:, self.feet_indices, :2]  # [num_envs, 2, 2]
        
        # 将脚部位置转换为地形网格坐标
        feet_grid_pos = (feet_pos + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale
        feet_grid_pos = feet_grid_pos.long()
        
        # 确保坐标在有效范围内
        feet_grid_pos[..., 0] = torch.clamp(feet_grid_pos[..., 0], 0, self.terrain.valid_standing_mask.shape[0]-1)
        feet_grid_pos[..., 1] = torch.clamp(feet_grid_pos[..., 1], 0, self.terrain.valid_standing_mask.shape[1]-1)
        
        # 检查脚部是否在可站立区域
        feet_in_valid_area = self.terrain.valid_standing_mask[feet_grid_pos[..., 0], feet_grid_pos[..., 1]]
        
        # 只对接触地面的脚部进行惩罚
        contact_mask = self.contact_filt  # [num_envs, 2]
        
        # 计算惩罚：接触地面但不在可站立区域的脚部
        penalty_mask = contact_mask & (~feet_in_valid_area.bool())
        
        # 考虑脚部大小：如果脚部部分在可站立区域，减少惩罚
        foot_size_consideration = self._consider_foot_size(feet_pos, feet_grid_pos)
        penalty_mask = penalty_mask & (~foot_size_consideration)
        
        # 只对在parkour区域内的机器人应用惩罚
        penalty_mask = penalty_mask & in_parkour_area.unsqueeze(1)
        
        # 计算惩罚值
        penalty = torch.sum(penalty_mask.float(), dim=1)
        
        return penalty

    def _consider_foot_size(self, feet_pos, feet_grid_pos):
        """考虑脚部大小，避免误判"""
        foot_radius = self.cfg.terrain.foot_size_tolerance  # 脚部半径（米）
        foot_radius_grid = int(foot_radius / self.terrain.cfg.horizontal_scale)
        
        # 检查脚部周围区域是否有可站立点
        valid_neighbors = torch.zeros_like(feet_grid_pos[..., 0], dtype=torch.bool)
        
        for dx in range(-foot_radius_grid, foot_radius_grid + 1):
            for dy in range(-foot_radius_grid, foot_radius_grid + 1):
                if dx*dx + dy*dy <= foot_radius_grid*foot_radius_grid:
                    neighbor_x = torch.clamp(feet_grid_pos[..., 0] + dx, 0, self.terrain.valid_standing_mask.shape[0]-1)
                    neighbor_y = torch.clamp(feet_grid_pos[..., 1] + dy, 0, self.terrain.valid_standing_mask.shape[1]-1)
                    valid_neighbors |= self.terrain.valid_standing_mask[neighbor_x, neighbor_y].bool()
        
        return valid_neighbors

    def _prepare_reward_function(self):
        """重写奖励函数准备方法，添加parkour相关奖励"""
        # 调用基类方法
        super()._prepare_reward_function()
        
        # 添加parkour惩罚奖励（仅在第一阶段训练时）
        if (hasattr(self.cfg.terrain, 'two_stage_training') and 
            self.cfg.terrain.two_stage_training and 
            self.cfg.terrain.training_stage == 1):
            
            # 添加parkour惩罚到奖励函数列表
            if hasattr(self.cfg.rewards.scales, 'feet_parkour_penalty'):
                self.reward_functions.append(self._reward_feet_parkour_penalty)
                self.reward_names.append('feet_parkour_penalty')
                
                # 更新episode_sums
                self.episode_sums['feet_parkour_penalty'] = torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
                )

    
#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, local_key_body_pos, dof_offsets):
    local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
    local_root_vel = quat_rotate_inverse(root_rot, root_vel)
    # print(local_root_vel[0])

    # heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    # local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)
    # local_root_vel = quat_rotate(heading_rot, root_vel)
    # print(local_root_vel[0], "\n")

    # root_pos_expand = root_pos.unsqueeze(-2)  # [num_envs, 1, 3]
    # local_key_body_pos = key_body_pos - root_pos_expand
    
    # heading_rot_expand = heading_rot.unsqueeze(-2)
    # heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    # flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    # flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])
    # local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    # flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    roll, pitch, yaw = euler_from_quaternion(root_rot)
    return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)

@torch.jit.script
def reindex_motion_dof(dof, indices_sim, indices_motion, valid_dof_body_ids):
    dof = dof.clone()
    dof[:, indices_sim] = dof[:, indices_motion]
    return dof[:, valid_dof_body_ids]

@torch.jit.script
def local_to_global(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = rigid_body_pos.reshape(total_bodies, 3)
    global_body_pos = quat_rotate(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3) + root_pos[:, None, :3]
    return global_body_pos

@torch.jit.script
def global_to_local(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = (rigid_body_pos - root_pos[:, None, :3]).view(total_bodies, 3)
    local_end_pos = quat_rotate_inverse(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3)
    return local_end_pos

@torch.jit.script
def global_to_local_xy(yaw, global_pos_delta):
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    rotation_matrices = torch.stack([cos_yaw, sin_yaw, -sin_yaw, cos_yaw], dim=2).view(-1, 2, 2)
    local_pos_delta = torch.bmm(rotation_matrices, global_pos_delta.unsqueeze(-1))
    return local_pos_delta.squeeze(-1)