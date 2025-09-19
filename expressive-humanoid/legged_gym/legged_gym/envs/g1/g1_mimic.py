from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse

import torch, torchvision

from legged_gym import LEGGED_GYM_ROOT_DIR, ASE_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from legged_gym.utils.math import *
# 移除错误的导入，使用标准的torch.rand
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

class G1Mimic(LeggedRobot):
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
        
        # 初始化ASE location goals相关变量
        self._tar_change_steps = None
        self._tar_change_counter = None
        self._tar_pos = None
        self._tar_dir = None
        
        # Pre init for motion loading (与H1保持一致)
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        
        self.init_motions(cfg)
        if cfg.motion.num_envs_as_motions:
            self.cfg.env.num_envs = 1  # G1暂时没有motion数据，保持原配置
        
        # 确保num_actions正确设置为G1的12个DOF
        self.cfg.env.num_actions = self.cfg.env.num_policy_actions
        
        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        
        # 初始化ASE Location Goals缓冲区
        if self.use_ase_location_goals:
            self._init_ase_location_goals()
        
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

        self.init_motion_buffers(cfg)

        self.reset_idx(torch.arange(self.num_envs, device=self.device), init=True)
        self.post_physics_step()
    
    def _parse_cfg(self, cfg):
        """解析配置参数"""
        super()._parse_cfg(cfg)
        
        # 从配置中读取ASE相关参数
        self.use_ase_location_goals = getattr(cfg.env, 'use_ase_location_goals', False)
        self._tar_speed = getattr(cfg.env, 'tar_speed', 1.0)
        self._tar_change_steps_min = getattr(cfg.env, 'tar_change_steps_min', 100)
        self._tar_change_steps_max = getattr(cfg.env, 'tar_change_steps_max', 300)
    
    def _init_motion_lib(self):
        """初始化motion library"""
        # G1暂时没有motion数据，先跳过motion library初始化
        # 后续有retarget数据后再实现
        pass
    
    def _init_g1_buffers(self):
        """初始化G1特定的缓冲区"""
        # 初始化缺失的属性，确保与H1兼容
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.delta_next_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.target_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.next_target_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # 初始化ASE location goals相关变量
        if self.use_ase_location_goals:
            self.ase_delta_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            self._goal_reach_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            self._prev_root_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
    
    def _reset_ase_location_goals(self, env_ids):
        """重置ASE风格的location goals"""
        if not self.use_ase_location_goals:
            return
            
        # 为指定环境生成新的目标
        new_tar_pos = torch.randn(len(env_ids), 2, device=self.device) * 5.0
        new_tar_dir = torch.randn(len(env_ids), 2, device=self.device)
        new_tar_dir = new_tar_dir / torch.norm(new_tar_dir, dim=1, keepdim=True)
        
        self._tar_pos[env_ids] = new_tar_pos
        self._tar_dir[env_ids] = new_tar_dir
        
        # 重置计数器
        self._tar_change_counter[env_ids] = 0
        self._tar_change_steps[env_ids] = torch.randint(
            self._tar_change_steps_min, 
            self._tar_change_steps_max + 1, 
            (len(env_ids),), 
            device=self.device
        )
        
        # 重置目标到达计时器
        self._goal_reach_timer[env_ids] = 0.0
    
    def _init_ase_location_goals(self):
        """初始化ASE风格的location goals"""
        if not self.use_ase_location_goals:
            return
            
        # 初始化目标变化步数
        self._tar_change_steps = torch.randint(
            self._tar_change_steps_min, 
            self._tar_change_steps_max + 1, 
            (self.num_envs,), 
            device=self.device
        )
        self._tar_change_counter = torch.zeros(self.num_envs, device=self.device)
        
        # 初始化目标位置和方向
        self._tar_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._tar_dir = torch.zeros(self.num_envs, 2, device=self.device)
        
        # 设置初始目标
        self._update_ase_location_goals()
    
    def _update_ase_location_goals(self):
        """更新ASE风格的location goals"""
        if not self.use_ase_location_goals:
            return
            
        # 更新目标变化计数器
        self._tar_change_counter += 1
        
        # 检查是否需要更新目标
        need_update = self._tar_change_counter >= self._tar_change_steps
        
        if need_update.any():
            # 为需要更新的环境生成新的目标
            new_tar_pos = torch.randn(self.num_envs, 2, device=self.device) * 5.0
            new_tar_dir = torch.randn(self.num_envs, 2, device=self.device)
            new_tar_dir = new_tar_dir / torch.norm(new_tar_dir, dim=1, keepdim=True)
            
            # 只更新需要更新的环境
            self._tar_pos[need_update] = new_tar_pos[need_update]
            self._tar_dir[need_update] = new_tar_dir[need_update]
            
            # 重置计数器
            self._tar_change_counter[need_update] = 0
            self._tar_change_steps[need_update] = torch.randint(
                self._tar_change_steps_min, 
                self._tar_change_steps_max + 1, 
                (need_update.sum(),), 
                device=self.device
            )
        
        # 计算ASE goal的朝向差
        ase_goal_rel = self._tar_pos - self.root_states[:, :2]
        norm = torch.norm(ase_goal_rel, dim=-1, keepdim=True)
        ase_goal_vec_norm = ase_goal_rel / (norm + 1e-5)
        self.ase_goal_yaw = torch.atan2(ase_goal_vec_norm[:, 1], ase_goal_vec_norm[:, 0])
        self.ase_delta_yaw = self.ase_goal_yaw - self.yaw
    
    def _update_yaw_tracking(self):
        """更新yaw相关变量"""
        # 从base orientation提取当前yaw
        quat = self.base_quat
        self.yaw = torch.atan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                              1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
        
        # 保持目标为零（没有特定的朝向目标）
        self.target_yaw.fill_(0.0)
        self.next_target_yaw.fill_(0.0)
    
    def _get_noise_scale_vec(self, cfg):
        """与 H1Mimic 对齐的噪声向量布局，仅作用于本体感知段(n_proprio)。"""
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        noise_scale_vec[:, :3] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, 3:5] = self.cfg.noise.noise_scales.imu
        # 跳过 commands/goal 等段，与 H1 的索引保持一致，从第7位开始是关节量
        noise_scale_vec[:, 7:7+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, 7+self.num_dof:7+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        return noise_scale_vec

    def _init_foot(self):
        """初始化足部相关变量"""
        self.feet_num = len(self.feet_indices)
        
        # 确保rigid_body_states是三维的[num_envs, num_bodies, 13]
        assert len(self.rigid_body_states.shape) == 3, f"rigid_body_states should be 3D, got shape {self.rigid_body_states.shape}"
        assert self.rigid_body_states.shape[0] == self.num_envs, f"Expected {self.num_envs} envs, got {self.rigid_body_states.shape[0]}"
        assert self.rigid_body_states.shape[2] == 13, f"Expected 13 state dims, got {self.rigid_body_states.shape[2]}"
        
        # 直接使用基类已经设置好的三维状态张量
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
        # 确保足部状态维度正确
        assert self.feet_pos.shape == (self.num_envs, self.feet_num, 3), f"feet_pos shape mismatch: {self.feet_pos.shape}"
        assert self.feet_vel.shape == (self.num_envs, self.feet_num, 3), f"feet_vel shape mismatch: {self.feet_vel.shape}"
        
    def _init_buffers(self):
        """初始化缓冲区"""
        super()._init_buffers()
        # 先初始化G1特定缓冲，提供 target_yaw/next_target_yaw/yaw 等
        self._init_g1_buffers()
        # 再初始化足部状态视图
        self._init_foot()

    def update_feet_state(self):
        """更新足部状态"""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
    def _compute_observations(self):
        """计算观测值"""
        # 更新ASE location goals
        if self.use_ase_location_goals:
            self._update_ase_location_goals()
        
        # 调用父类的观测计算
        super()._compute_observations()

    def _compute_reward(self):
        """计算奖励"""
        # 更新ASE location goals
        if self.use_ase_location_goals:
            self._update_ase_location_goals()
        
        # 调用父类的奖励计算
        super()._compute_reward()
    
    def post_physics_step(self):
        """重写post_physics_step，确保yaw更新"""
        # 更新yaw值
        self._update_yaw_tracking()
        # 调用父类实现
        super().post_physics_step()

    def reindex(self, vec):
        """G1的reindex方法，与H1保持一致但适配12个DOF"""
        # G1有12个DOF，直接返回原向量
        return vec
    
    def reindex_feet(self, vec):
        """G1的足部reindex方法"""
        # G1有2个足部，直接返回原向量
        return vec
    
    def reindex_dof_pos_vel(self, dof_pos, dof_vel):
        """G1的DOF重索引方法，暂时直接返回"""
        # G1暂时没有motion数据，直接返回原值
        return dof_pos, dof_vel
    
    def compute_obs_buf(self):
        """计算观测缓冲区，与H1保持一致但适配G1的12个DOF"""
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        
        if self.use_ase_location_goals:
            # ASE Location Goals观测
            goal_obs1 = 0 * self.ase_delta_yaw[:, None]
            goal_obs2 = self.ase_delta_yaw[:, None]
            goal_obs3 = self.ase_delta_yaw[:, None]
        else:
            # Terrain Goals观测
            if hasattr(self, 'cur_goals') and self.cur_goals is not None:
                cur_goal_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
                norm = torch.norm(cur_goal_rel, dim=-1, keepdim=True)
                cur_goal_vec_norm = cur_goal_rel / (norm + 1e-5)
                cur_goal_yaw = torch.atan2(cur_goal_vec_norm[:, 1], cur_goal_vec_norm[:, 0])
                terrain_delta_yaw = cur_goal_yaw - self.yaw
                
                if hasattr(self, 'next_goals') and self.next_goals is not None:
                    next_goal_rel = self.next_goals[:, :2] - self.root_states[:, :2]
                    norm = torch.norm(next_goal_rel, dim=-1, keepdim=True)
                    next_goal_vec_norm = next_goal_rel / (norm + 1e-5)
                    next_goal_yaw = torch.atan2(next_goal_vec_norm[:, 1], next_goal_vec_norm[:, 0])
                    terrain_delta_next_yaw = next_goal_yaw - self.yaw
                else:
                    terrain_delta_next_yaw = terrain_delta_yaw
            else:
                terrain_delta_yaw = torch.zeros_like(self.yaw)
                terrain_delta_next_yaw = torch.zeros_like(self.yaw)
            
            goal_obs1 = 0 * terrain_delta_yaw[:, None]
            goal_obs2 = terrain_delta_yaw[:, None]
            goal_obs3 = terrain_delta_next_yaw[:, None]
            
        return torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,   # [3] 角速度
            imu_obs,    # [2] roll, pitch
            goal_obs1,  # [1] 置零的朝向差
            goal_obs2,  # [1] 当前目标朝向差
            goal_obs3,  # [1] 下一个目标朝向差
            0*self.commands[:, 0:2],  # [2] 置零的命令
            self.commands[:, 0:1],    # [1] 线速度命令
            (self.env_class != 17).float()[:, None],  # [1] 环境类别标志1
            (self.env_class == 17).float()[:, None],  # [1] 环境类别标志2
            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),  # [12] DOF位置
            self.reindex(self.dof_vel * self.obs_scales.dof_vel),                              # [12] DOF速度
            self.reindex(self.action_history_buf[:, -1]),                                     # [12] 上一动作
            self.reindex_feet(self.contact_filt.float()*0-0.5),                               # [2] 接触状态
        ), dim=-1)
    
    def compute_obs_demo(self):
        """计算demo观测，暂时返回零值"""
        return torch.zeros((self.num_envs, self.cfg.env.n_demo), device=self.device)
    
    def compute_observations(self):
        """计算观测值，与H1保持一致"""
        obs_buf = self.compute_obs_buf()

        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_scale
        
        obs_demo = self.compute_obs_demo()
        
        motion_features = self.obs_history_buf[:, -self.cfg.env.prop_hist_len:].flatten(start_dim=1)
        priv_explicit = torch.cat((0*self.base_lin_vel * self.obs_scales.lin_vel,), dim=-1)
        
        # 安全地使用motor_strength
        try:
            priv_latent = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1, 
                self.motor_strength[1] - 1
            ), dim=-1)
        except Exception as e:
            print(f"[ERROR] Failed to use motor_strength in priv_latent: {e}")
            fallback_motor_0 = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
            fallback_motor_1 = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
            
            priv_latent = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                fallback_motor_0, 
                fallback_motor_1
            ), dim=-1)
        
        # 获取 scan 观测
        scan_obs = None
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            scan_obs = heights
            self.obs_buf = torch.cat([motion_features, obs_buf, obs_demo, scan_obs, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            n_scan = getattr(self.cfg, 'n_scan', 132)
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

        contact_filt_safe = self.contact_filt.float()
        if contact_filt_safe.dim() != 2 or contact_filt_safe.shape[0] != self.num_envs:
            contact_filt_safe = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        
        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([contact_filt_safe] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                contact_filt_safe.unsqueeze(1)
            ], dim=1)
        )
    
    def update_demo_obs(self):
        """更新demo观测，暂时为空实现"""
        pass
    
    def check_termination(self):
        """检查终止条件，与H1保持一致"""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        
        # 早停机制
        if hasattr(self, 'cur_goals') and self.cur_goals is not None:
            robot_to_goal = self.cur_goals[:, :2] - self.root_states[:, :2]
            robot_velocity = self.base_lin_vel[:, :2]
            dot_product = torch.sum(robot_to_goal * robot_velocity, dim=1)
            distance_to_goal = torch.norm(robot_to_goal, dim=1)
            
            early_stop = (distance_to_goal < self.cfg.env.early_stop_distance_threshold) & \
                        (dot_product < self.cfg.env.early_stop_velocity_threshold)
            self.reset_buf |= early_stop

        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
    
    def fetch_amp_obs_demo(self, num_samples):
        """无数据模式：返回空张量，满足AMP取样接口。

        Args:
            num_samples (int): 需要的样本数量

        Returns:
            torch.Tensor: 形状 [0, amp_input_dim] 的空张量
        """
        # 与 G1MimicAMPCfgPPO.amp.amp_input_dim 对齐。此处避免循环依赖，直接计算：
        num_obs_steps = 10
        num_obs_per_step = 39
        amp_input_dim = num_obs_steps * num_obs_per_step
        return torch.empty(0, amp_input_dim, device=self.device)
    
    ######### Rewards #########
    def compute_reward(self):
        """计算奖励，与H1保持一致"""
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
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
    
    def _reward_next_goal_direction(self):
        """奖励机器人朝向下一个目标的方向"""
        if self.use_ase_location_goals:
            # ASE Location Goals方向奖励
            tar_dir = self._tar_pos - self.root_states[:, :2]
            tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
            
            heading_rot = self.base_quat
            facing_dir = torch.zeros_like(self.root_states[:, :3])
            facing_dir[:, 0] = 1.0
            facing_dir = quat_apply(heading_rot, facing_dir)
            
            facing_err = torch.sum(tar_dir * facing_dir[:, :2], dim=-1)
            rew = torch.clamp_min(facing_err, 0.0)
            return rew
        else:
            # Terrain Goals方向奖励
            if hasattr(self, 'cur_goals') and self.cur_goals is not None:
                next_goal_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
                forward_vec = quat_apply(self.base_quat, torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, 3))
                
                goal_angle = torch.atan2(next_goal_rel[:, 1], next_goal_rel[:, 0])
                robot_angle = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])
                
                angle_diff = torch.abs(torch.atan2(torch.sin(goal_angle - robot_angle), torch.cos(goal_angle - robot_angle)))
                rew = torch.exp(-2.0 * angle_diff)
                return rew
            else:
                return torch.zeros(self.num_envs, device=self.device)

    def _reward_next_goal_distance(self):
        """奖励机器人接近下一个目标"""
        if self.use_ase_location_goals:
            distance_to_goal = torch.norm(self._tar_pos - self.root_states[:, :2], dim=1)
            rew = torch.exp(-0.5 * distance_to_goal)
            return rew
        else:
            if hasattr(self, 'cur_goals') and self.cur_goals is not None:
                distance_to_next = torch.norm(self.cur_goals[:, :2] - self.root_states[:, :2], dim=1)
                rew = torch.exp(-0.5 * distance_to_next)
                return rew
            else:
                return torch.zeros(self.num_envs, device=self.device)

    def _reward_next_goal_velocity(self):
        """奖励机器人以合适速度朝向下一个目标移动"""
        if self.use_ase_location_goals:
            tar_dir = self._tar_pos - self.root_states[:, :2]
            tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
            
            root_vel = self.root_states[:, 7:10]
            tar_dir_speed = torch.sum(tar_dir * root_vel[:, :2], dim=-1)
            
            tar_speed = self._tar_speed
            tar_vel_err = tar_speed - tar_dir_speed
            tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
            
            vel_reward = torch.exp(-4.0 * (tar_vel_err * tar_vel_err))
            speed_mask = tar_dir_speed <= 0
            vel_reward[speed_mask] = 0
            
            distance_to_goal = torch.norm(self._tar_pos - self.root_states[:, :2], dim=1)
            dist_mask = distance_to_goal < self._goal_reach_threshold
            vel_reward[dist_mask] = 1.0
            
            return vel_reward
        else:
            if hasattr(self, 'cur_goals') and self.cur_goals is not None:
                tar_dir = self.cur_goals[:, :2] - self.root_states[:, :2]
                tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
                
                root_vel = self.root_states[:, 7:10]
                tar_dir_speed = torch.sum(tar_dir * root_vel[:, :2], dim=-1)
                
                tar_speed = 1.0
                tar_vel_err = tar_speed - tar_dir_speed
                tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
                
                vel_reward = torch.exp(-4.0 * (tar_vel_err * tar_vel_err))
                speed_mask = tar_dir_speed <= 0
                vel_reward[speed_mask] = 0
                
                distance_to_next = torch.norm(self.cur_goals[:, :2] - self.root_states[:, :2], dim=1)
                dist_mask = distance_to_next < 0.5
                vel_reward[dist_mask] = 1.0
                
                return vel_reward
            else:
                return torch.zeros(self.num_envs, device=self.device)

    def _reward_feet_drag(self):
        """奖励足部拖拽"""
        feet_xyz_vel = torch.abs(self.rigid_body_states[:, self.feet_indices, 7:10]).sum(dim=-1)
        dragging_vel = self.contact_filt * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        return rew
    
    def _reward_energy(self):
        """奖励能量消耗"""
        return torch.norm(torch.abs(self.torques * self.dof_vel), dim=-1)

    def _reward_feet_air_time(self):
        """奖励足部空中时间"""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_feet_height(self):
        """奖励足部高度"""
        feet_height = self.rigid_body_states[:, self.feet_indices, 2]
        rew = torch.clamp(torch.norm(feet_height, dim=-1) - 0.2, max=0)
        return rew
    
    def _reward_feet_force(self):
        """奖励足部力"""
        rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
        rew[rew < 500] = 0
        rew[rew > 500] -= 500
        return rew

    def _reward_dof_error(self):
        """奖励DOF误差"""
        # G1 共有 12 个 DOF，全量计入误差
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, :12], dim=1)
        return dof_error

    def _reward_dof_pos_limits(self):
        """靠近关节极限惩罚（与H1保持一致）"""
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_parkour_penalty(self):
        """惩罚脚部接触非parkour区域（仅在第一阶段训练时启用）"""
        if not (hasattr(self.cfg.terrain, 'two_stage_training') and 
                self.cfg.terrain.two_stage_training and 
                self.cfg.terrain.training_stage == 1):
            return torch.zeros(self.num_envs, device=self.device)

        if not hasattr(self.terrain, 'valid_standing_mask'):
            return torch.zeros(self.num_envs, device=self.device)

        robot_pos = self.root_states[:, :2]

        parkour_start_x = 20.0
        parkour_start_y = -2.0
        parkour_end_y = 2.0

        penalty = torch.zeros(self.num_envs, device=self.device)

        for env_id in range(self.num_envs):
            x = robot_pos[env_id, 0]
            y = robot_pos[env_id, 1]

            if x >= parkour_start_x:
                if (y < parkour_start_y) or (y > parkour_end_y):
                    penalty[env_id] = 1.0
        return penalty
    
    ######### 缺失的关键方法 #########
    def step(self, actions):
        """重写step方法，与H1保持一致"""
        actions = self.reindex(actions)
        actions.to(self.device)
        
        # 记录action历史
        if hasattr(self, 'action_history_buf'):
            self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        
        # G1的ankle限制（适配12DOF）
        # G1的ankle joint indices: [4, 5, 10, 11] (ankle_pitch, ankle_roll for both legs)
        ankle_indices = [4, 5, 10, 11]
        for idx in ankle_indices:
            if idx < actions.shape[1]:
                self.actions[:, idx] = torch.clamp(actions[:, idx], -0.5, 0.5)
        
        # 其他动作不限制
        for i in range(actions.shape[1]):
            if i not in ankle_indices:
                self.actions[:, i] = actions[:, i]
        
        self.global_counter += 1
        self.total_env_steps_counter += 1
        
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(self.actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def reset_idx(self, env_ids, init=False):
        """重置指定环境，与H1保持一致但适配G1"""
        if len(env_ids) == 0:
            return
        
        # 更新课程学习完成率
        completion_rate_mean = 0.0
        
        # G1暂时没有motion数据，使用默认DOF位置
        dof_pos_default = self.default_dof_pos + (2.0 * torch.rand((len(env_ids), self.num_dof), device=self.device) - 1.0) * 0.2 * self.default_dof_pos
        dof_vel_default = torch.zeros_like(dof_pos_default)
        
        # 更新地形课程学习
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # 重置机器人状态
        self._reset_dofs(env_ids, dof_pos_default, dof_vel_default)
        self._reset_root_states(env_ids)

        if init:
            self.init_root_pos_global = self.root_states[:, :3].clone()
            self.target_pos_abs = self.init_root_pos_global.clone()[:, :2]
        else:
            self.init_root_pos_global[env_ids] = self.root_states[env_ids, :3].clone()
            self.target_pos_abs[env_ids] = self.init_root_pos_global[env_ids].clone()[:, :2]

        self._resample_commands(env_ids)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 重置缓冲区
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        if hasattr(self, 'obs_history_buf'):
            self.obs_history_buf[env_ids, :, :] = 0.
        if hasattr(self, 'contact_buf'):
            self.contact_buf[env_ids, :, :] = 0.
        if hasattr(self, 'action_history_buf'):
            self.action_history_buf[env_ids, :, :] = 0.
        self.last_contacts[env_ids] = 0.
        
        # 重置goals
        if self.use_ase_location_goals:
            self._reset_ase_location_goals(env_ids)
            if hasattr(self, '_prev_root_pos'):
                self._prev_root_pos[env_ids] = self.root_states[env_ids, :3]
        else:
            if hasattr(self, 'cur_goal_idx'):
                self.cur_goal_idx[env_ids] = 0
            if hasattr(self, 'reach_goal_timer'):
                self.reach_goal_timer[env_ids] = 0

        # 填充extras
        self.extras["episode"] = {}
        self.extras["episode"]["curriculum_completion"] = completion_rate_mean
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        return
    
    def _reset_dofs(self, env_ids, dof_pos, dof_vel):
        """重置DOF状态"""
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def init_motions(self, cfg):
        """初始化motion相关参数（G1版本）"""
        # G1暂时没有motion数据，初始化基本结构
        self._key_body_ids = torch.tensor([3, 6], device=self.device)  # G1的关键body ids
        self._key_body_ids_sim = torch.tensor([1, 4, 6, 9], device=self.device)  # G1的sim body ids
        self._key_body_ids_sim_subset = torch.tensor([0, 1, 2, 3], device=self.device)
        
        self._num_key_bodies = len(self._key_body_ids_sim_subset)
        
        # G1的DOF索引（12个DOF，直接映射）
        self.dof_indices_sim = torch.arange(12, device=self.device, dtype=torch.long)
        self.dof_indices_motion = torch.arange(12, device=self.device, dtype=torch.long)
        
        # G1暂时没有motion library，跳过
        pass
    
    def init_motion_buffers(self, cfg):
        """初始化motion buffers（G1版本）"""
        # G1暂时没有motion数据，初始化空buffers
        self._motion_dt = self.dt
        self._motion_num_future_steps = self.cfg.env.n_demo_steps
        self._motion_demo_offsets = torch.arange(0, self.cfg.env.n_demo_steps * self.cfg.env.interval_demo_steps, self.cfg.env.interval_demo_steps, device=self.device)
        self._demo_obs_buf = torch.zeros((self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo), device=self.device)
        self._curr_demo_obs_buf = self._demo_obs_buf[:, 0, :]
        self._next_demo_obs_buf = self._demo_obs_buf[:, 1, :]

        # 初始化其他必要的缓冲区
        self.global_counter = 0
        self.total_env_steps_counter = 0
    
    def _resample_commands(self, env_ids):
        """重新采样命令"""
        self.commands[env_ids, :] = 0


#####################################################################
###=========================jit functions=========================###
#####################################################################

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

def local_to_global(quat, local_pos, root_pos):
    num_key_bodies = local_pos.shape[1]
    num_envs = local_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_local_pos = local_pos.view(total_bodies, 3)
    flat_global_pos = quat_rotate(flat_heading_rot, flat_local_pos).view(num_envs, num_key_bodies, 3)
    global_pos = flat_global_pos + root_pos[:, None, :3]
    return global_pos