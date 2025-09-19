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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1MimicCfg( LeggedRobotCfg ):
    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y up, 1 is z up
        use_gpu = True
        physics_engine = "physx"
        physx = {
            "num_threads": 4,
            "solver_type": 1,
            "num_position_iterations": 8,  # 增加位置迭代次数
            "num_velocity_iterations": 2,  # 增加速度迭代次数
            "contact_offset": 0.02,  # 增加接触偏移量
            "rest_offset": 0.0,
            "bounce_threshold_velocity": 0.2,  # 降低反弹阈值
            "max_depenetration_velocity": 10.0,  # 增加最大去穿透速度
            "default_friction": 0.5,  # 添加默认摩擦系数
            "friction_offset": 0.04,  # 添加摩擦偏移量
            "max_gpu_contact_pairs": 8388608,
            "default_buffer_size_multiplier": 5.0,
            "contact_collection": 2  # 0: CC_NEVER (1.3.4), 1: CC_LAST_SUBSTEP (1.3.4), 2: CC_ALL_SUBSTEPS (1.3.4) [default = 2]
        }
    class env( LeggedRobotCfg.env ):
        num_envs = 6144

        n_demo_steps = 2
        n_demo = 9 + 3 + 3 + 3 +6*3  #observe height
        interval_demo_steps = 0.1

        n_scan = 132  # 启用132维scan观测
        n_priv = 3
        n_priv_latent = 4 + 1 + 12*2  # G1有12个DOF
        n_proprio = 3 + 2 + 3 + 3 + 2 + 12*3 + 2 # 角速度3 + IMU2 + goal3 + 命令3 + 环境类别2 + DOF相关36 + 接触2 = 51
        history_len = 10

        prop_hist_len = 4
        n_feature = prop_hist_len * n_proprio

        num_observations = n_feature + n_proprio + n_demo + n_scan + history_len*n_proprio + n_priv_latent + n_priv

        episode_length_s = 50 # episode length in seconds
        
        # 早停机制参数
        early_stop_distance_threshold = 5  # 距离目标的阈值（米）
        early_stop_velocity_threshold = -10  # 速度方向阈值（负值表示背离目标）
        enable_mimic_dev_termination = False  # disable termination due to mimic deviation
        min_forward_disp_8s = 1.0  # meters; early stop if forward displacement in 8s is below this
        # 新增：未达下一个目标点的时间阈值（秒），用于早停验证
        no_next_goal_time_s = 8.0
        
        num_policy_actions = 12  # G1有12个DOF
        
        # 添加goal相关的配置
        next_goal_threshold = 0.2  # 到达goal的阈值
        num_future_goal_obs = 1    # 未来goal观测的数量
        reach_goal_delay = 0.1     # 到达goal后的延迟时间

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.80]  # x,y,z [m] - 降低初始高度以适应G1
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,
           'left_hip_roll_joint' : 0,
           'left_hip_pitch_joint' : -0.2,  # 增大hip pitch使姿态更稳定
           'left_knee_joint' : 0.4,        # 增大knee弯曲
           'left_ankle_pitch_joint' : -0.2,
           'left_ankle_roll_joint' : 0,
           'right_hip_yaw_joint' : 0.,
           'right_hip_roll_joint' : 0,
           'right_hip_pitch_joint' : -0.2,  # 对称设置
           'right_knee_joint' : 0.4,        # 对称设置
           'right_ankle_pitch_joint': -0.2,
           'right_ankle_roll_joint' : 0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]
        action_scale = 0.25
        decimation = 4

    class normalization( LeggedRobotCfg.normalization):
        clip_actions = 10

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof_with_hand.urdf'
        name = "g1_fix_upper"
        torso_name = "pelvis"  # 这个参数是必需的
        foot_name = "ankle_roll"
        knee_name = "knee"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # tracking rewards
            alive = 1
            # tracking_demo_goal_vel = 1.0
            # tracking_mul = 6
           # tracking_lin_vel = 6
            # stand_still = 3
            # tracking_goal_vel = 4

            # 新的next goal相关奖励
            next_goal_direction = 3    # 朝向next goal的奖励权重
            next_goal_distance = 2     # 接近next goal的奖励权重
            next_goal_velocity = 4     # 朝向下一个目标的速度奖励权重
            

           # tracking_demo_yaw = 1
           # tracking_demo_roll_pitch = 1
            orientation = -1
           # tracking_demo_dof_pos = 3  # 注释掉原来的DOF跟踪奖励
            #tracking_demo_lower_body_dof_pos = 3  # 新增下半身DOF跟踪奖励权重
            # tracking_demo_dof_vel = 1.0
            # tracking_demo_key_body = 2
            # tracking_demo_height = 1  # useful if want better height tracking
            
            # tracking_demo_lin_vel = 1
            # tracking_demo_ang_vel = 0.5
            # regularization rewards
            lin_vel_z = -1.0
           # ang_vel_xy = -0.4
            # orientation = -1.
            #dof_acc = -3e-7
            collision = -10.
            action_rate = -0.1
            # delta_torques = -1.0e-7
           # torques = -1e-5
            energy = -1e-3
            #hip_pos = -0.5
           # dof_error = -0.1
            feet_stumble = -1
            feet_edge = -1
            feet_drag = - 1
            dof_pos_limits = -10.0
           # feet_air_time = 10
           # feet_height = 2
            feet_force = -3e-3
            
            # parkour训练相关奖励
            feet_parkour_penalty = -5.0  # 脚部接触非parkour区域的严重惩罚权重

        only_positive_rewards = False
        clip_rewards = True
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized
        is_play = False

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_gravity = True
        gravity_rand_interval_s = 10
        gravity_range = [-0.1, 0.1]

    class noise():
        add_noise = True
        noise_scale = 0.5 # scales other values
        class noise_scales():
            dof_pos = 0.01
            dof_vel = 0.15
            ang_vel = 0.3
            imu = 0.2

    class amp():
        num_obs_steps = 10
        num_obs_per_step = 12 + 3 # 12 joint angles + 3 base ang vel

    class motion:
        motion_curriculum = True
        motion_type = "yaml"
        motion_name = "motions_autogen_all_no_run_jump.yaml"  # 镜像H1的motion文件

        global_keybody = False
        global_keybody_reset_time = 2

        num_envs_as_motions = False

        no_keybody = False
        regen_pkl = False

        step_inplace_prob = 0.05
        resample_step_inplace_interval_s = 10


    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "trimesh"
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        num_goals = 5  # 匹配challenging_terrain的配置
        
        # 添加challenging_terrain需要的配置
        border_size = 5
        horizontal_scale = 0.05
        vertical_scale = 0.005
        
        # 两阶段训练配置
        two_stage_training = True
        training_stage = 1  # 1: 平地训练, 2: 真实地形训练
        stage1_duration = 1000000  # 阶段一训练步数
        stage2_duration = 1000000  # 阶段二训练步数
        foot_size_tolerance = 0.1  # 脚部大小容忍度（米）

class G1MimicCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimic"
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPOMimic'
    
    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = False
        text_feat_input_dim = G1MimicCfg.env.n_feature
        text_feat_output_dim = 16
        feat_hist_len = G1MimicCfg.env.prop_hist_len
        # actor_hidden_dims = [1024, 512]
        # critic_hidden_dims = [1024, 512]
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005

    class estimator:
        train_with_estimated_states = False
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = G1MimicCfg.env.n_priv
        priv_start = G1MimicCfg.env.n_feature + G1MimicCfg.env.n_proprio + G1MimicCfg.env.n_demo + G1MimicCfg.env.n_scan
        
        prop_start = G1MimicCfg.env.n_feature
        prop_dim = G1MimicCfg.env.n_proprio


class G1MimicDistillCfgPPO( G1MimicCfgPPO ):
    class distill:
        num_demo = 3
        num_steps_per_env = 24
        
        num_pretrain_iter = 0

        activation = "elu"
        learning_rate = 1.e-4
        student_actor_hidden_dims = [1024, 1024, 512]

        num_mini_batches = 4