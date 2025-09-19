from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G1Robot(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        
        # Initialize missing attributes for base class compute_observations
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.delta_next_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.target_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.next_target_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
    def _update_yaw_tracking(self):
        """ Update yaw-related variables for observation computation """
        # Extract current yaw from base orientation
        quat = self.base_quat
        self.yaw = torch.atan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                              1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
        
        # For visualization, keep targets at zero (no specific heading goal)
        self.target_yaw.fill_(0.0)
        self.next_target_yaw.fill_(0.0)
        
    def post_physics_step(self):
        """ Override to update yaw tracking before observations """
        # Update yaw values before parent's post_physics_step
        self._update_yaw_tracking()
        # Call parent implementation
        super().post_physics_step()
        
    # Using base class compute_observations method to avoid missing delta_yaw attribute issues
    # def _compute_observations(self):
    #     """ Computes observations
    #     """
    #     self.obs_buf = torch.cat((  self.base_ang_vel * self.obs_scales.ang_vel,
    #                                 self.projected_gravity,
    #                                 self.commands[:, :3] * self.commands_scale,
    #                                 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    #                                 self.dof_vel * self.obs_scales.dof_vel,
    #                                 self.actions
    #                                 ),dim=-1)

    def _compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode rewards and logs the reward terms
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def fetch_amp_obs_demo(self, num_samples):
        """ Placeholder method for AMP demo fetching - returns empty tensor for visualization mode
        
        Args:
            num_samples (int): Number of demo samples to fetch
            
        Returns:
            torch.Tensor: Empty tensor to satisfy AMP initialization requirements
        """
        # For visualization only - return empty tensor with correct shape [0, 390]
        # The 390 comes from amp_input_dim in the config
        amp_input_dim = 390  # From G1CfgPPO.amp.amp_input_dim
        return torch.empty(0, amp_input_dim, device=self.device)