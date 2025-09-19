from isaacgym.torch_utils import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import torch
from legged_gym.utils.math import *
from legged_gym.envs.h1.h1_mimic import H1Mimic, global_to_local, local_to_global
from isaacgym import gymtorch, gymapi, gymutil

import torch_utils

class H1MimicEval(H1Mimic):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 确保reached_goal_ids属性存在
        if not hasattr(self, 'reached_goal_ids'):
            self.reached_goal_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def check_termination(self):
        print(f"[DEBUG] H1MimicEval.check_termination called, episode_length_buf: {self.episode_length_buf[0].item()}")
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        height_cutoff = self.root_states[:, 2] < 0.5

        # motion_end = self.episode_length_buf * self.dt >= self._motion_lengths
        # self.reset_buf |= motion_end

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # self.time_out_buf |= motion_end

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= height_cutoff
        
        # 添加早停逻辑（从H1Mimic.check_termination复制）
        # 调试信息
        self.extras["debug_cur_goals_exists"] = hasattr(self, 'cur_goals') and self.cur_goals is not None
        if hasattr(self, 'cur_goal_idx'):
            self.extras["debug_cur_goal_idx"] = self.cur_goal_idx[0].item()
        else:
            self.extras["debug_cur_goal_idx"] = -1
        self.extras["debug_episode_length"] = self.episode_length_buf[0].item()
        
        # 早停机制：如果机器人跑到当前目标前面太多，直接终止
        if hasattr(self, 'cur_goals') and self.cur_goals is not None:
            distance_to_goal = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1)
            velocity_to_goal = torch.sum(self.base_lin_vel[:, :2] * (self.cur_goals[:, :2] - self.root_states[:, :2]), dim=1)
            dot_product = velocity_to_goal / (torch.norm(self.base_lin_vel[:, :2], dim=1) + 1e-8)
            
            # 使用配置参数控制早停条件
            early_stop = (distance_to_goal < self.cfg.env.early_stop_distance_threshold) & \
                        (dot_product < self.cfg.env.early_stop_velocity_threshold)
            self.reset_buf |= early_stop

            # 未踩到下一个点直接早停：依据cur_goal_idx在过去no_next_goal_time_s是否递增
            try:
                window_s = float(getattr(self.cfg.env, 'no_next_goal_time_s', 8.0))
                window_steps = int(window_s / self.dt)
                if hasattr(self, '_last_goal_step') and hasattr(self, '_last_goal_idx') and window_steps > 0:
                    # 条件：当前未完成全部goals，且从_last_goal_step起步数差>=window_steps，且cur_goal_idx未变
                    not_finished = self.cur_goal_idx < self.cfg.terrain.num_goals
                    elapsed = (self.episode_length_buf - self._last_goal_step) >= window_steps
                    no_progress = self.cur_goal_idx == self._last_goal_idx
                    no_next_goal_window = not_finished & elapsed & no_progress
                    self.reset_buf |= no_next_goal_window
                    
                    # 调试信息
                    self.extras["debug_last_goal_step"] = self._last_goal_step[0].item()
                    self.extras["debug_elapsed_steps"] = (self.episode_length_buf - self._last_goal_step)[0].item()
                    self.extras["debug_window_steps"] = window_steps
                    self.extras["debug_no_next_goal_trigger"] = no_next_goal_window[0].item()
            except Exception as e:
                print(f"[DEBUG] Error in no_next_goal early stop: {e}")
                self.extras["debug_no_next_goal_error"] = str(e)
    
    # def resample_motion_times(self, env_ids):
    #     return 0*self._motion_lib.sample_time(self._motion_ids[env_ids])

    def render_record(self, mode="rgb_array"):
        if self.global_counter % 2 == 0:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            imgs = []
            for i in range(self.num_envs):
                cam = self._rendering_camera_handles[i]
                root_pos = self.root_states[i, :3].cpu().numpy()
                cam_pos = root_pos + np.array([0, -2, 0.3])
                self.gym.set_camera_location(cam, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
                
                img = self.gym.get_camera_image(self.sim, self.envs[i], cam, gymapi.IMAGE_COLOR)
                w, h = img.shape
                imgs.append(img.reshape([w, h // 4, 4]))
            return imgs
        return None
    
    def _create_envs(self):
        super()._create_envs()
        if self.cfg.env.record_video or self.cfg.env.record_frame:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 720
            camera_props.height = 480
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                # root_pos = self.root_states[i, :3].cpu().numpy()
                # cam_pos = root_pos + np.array([0, 1, 0.5])
                cam_pos = np.array([2, 0, 0.3])
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                self.gym.set_camera_location(camera_handle, self.envs[i], gymapi.Vec3(*cam_pos), gymapi.Vec3(*0*cam_pos))
    
    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reward_eval_ang_vel(self):
        return torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=-1))
            