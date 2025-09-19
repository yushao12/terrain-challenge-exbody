from isaacgym.torch_utils import *
import torch
from legged_gym.utils.math import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.g1.g1_mimic import G1Mimic, global_to_local, local_to_global
from legged_gym.envs.base.legged_robot import euler_from_quaternion

import torch_utils

class G1MimicAMP(G1Mimic):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        
        self._num_amp_obs_per_step = cfg.amp.num_obs_per_step
        self._num_amp_obs_steps = cfg.amp.num_obs_steps
        self._amp_obs_buf = torch.zeros((cfg.env.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=sim_device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        self._amp_obs_demo_buf = None

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    
    def reset_idx(self, env_ids, init=False):
        super().reset_idx(env_ids, init)
        if len(env_ids) != 0:
            self._compute_amp_observations(env_ids)
            self._init_amp_obs_default(env_ids)
        return
    
    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations() # latest on the left

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat
        return
    
    def _compute_amp_observations(self, env_ids=None):
        """在缺少运动/关键点数据时，直接输出零特征，保持维度对齐。"""
        if env_ids is None:
            self._curr_amp_obs_buf.zero_()
        else:
            self._curr_amp_obs_buf[env_ids].zero_()
        return
    
    ######### demonstrations #########
    def fetch_amp_obs_demo(self, num_samples):
        """G1暂时没有motion数据，返回空tensor"""
        num_obs_steps = 10
        num_obs_per_step = 39
        amp_input_dim = num_obs_steps * num_obs_per_step
        return torch.empty(0, amp_input_dim, device=self.device)
    
    ######### utils #########
    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
    
    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return
    
    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            # 安全地更新所有环境的观测历史
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                # 使用torch.clone()避免原地操作可能导致的索引问题
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i].clone()
        else:
            # 更新指定环境的观测历史
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i].clone()
        return
    
    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step


#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos):
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)
    local_root_vel = quat_rotate(heading_rot, root_vel)

    roll, pitch, yaw = euler_from_quaternion(root_rot)
    return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)

# @torch.jit.script
def build_amp_observations_curr(root_pos, root_rot, local_root_vel, local_root_ang_vel, dof_pos, dof_vel, local_key_body_pos):
    roll, pitch, yaw = euler_from_quaternion(root_rot)
    return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)