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
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER NOR THE NAMES OF ITS
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.g1.g1_mimic_config import G1MimicCfg, G1MimicCfgPPO

class G1MimicAMPCfg( G1MimicCfg ):
    class env( G1MimicCfg.env ):
        num_envs = 4096

    class motion:
        motion_curriculum = True
        motion_type = "yaml"
        motion_name = "motions_g1_all.yaml"  # 使用G1的重定向motion数据

        global_keybody = False
        global_keybody_reset_time = 2

        num_envs_as_motions = False

        no_keybody = False
        regen_pkl = False

        step_inplace_prob = 0.05
        resample_step_inplace_interval_s = 10

    class amp():
        num_obs_steps = 10
        # 简化观测：dof_pos(12) + local_root_vel(3) + local_root_ang_vel(3) + roll(1) + pitch(1) + root_height(1) + lower_body_key_pos(7*3)
        # G1有12个DOF和7个关键关节，所以是12 + 3 + 3 + 1 + 1 + 1 + 21 = 42
        num_obs_per_step = 12 + 3 + 3 + 1 + 1 + 1 + 7*3

class G1MimicAMPCfgPPO( G1MimicCfgPPO ):
    class runner( G1MimicCfgPPO.runner ):
        runner_class_name = "OnPolicyRunner"
        policy_class_name = 'ActorCriticRMA'
        algorithm_class_name = 'PPO'
    
    class amp():
        amp_input_dim = G1MimicAMPCfg.amp.num_obs_steps * G1MimicAMPCfg.amp.num_obs_per_step
        amp_disc_hidden_dims = [1024, 512]

        amp_replay_buffer_size = 1000000
        amp_demo_buffer_size = 200000
        amp_demo_fetch_batch_size = 512
        amp_learn_batch_size = 4096
        amp_learning_rate = 1.e-4

        amp_reward_coef = 4.0

        amp_grad_pen = 5