from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1ASELocationCfg(LeggedRobotCfg):
    class motion:
        motion_curriculum = True
        motion_type = "yaml"
        motion_name = "motions_easywalk.yaml"

        global_keybody = False
        global_keybody_reset_time = 2
        num_envs_as_motions = False
        no_keybody = False
        regen_pkl = False
        step_inplace_prob = 0.05
        resample_step_inplace_interval_s = 10

    class amp():
        num_obs_steps = 2
        num_obs_per_step = 19 + 3 + 3 + 3 + 12*3

    # ===== ASE Location Goals配置 =====
    class env:
        # ASE location goals参数
        tar_speed = 1.0  # 目标速度 (m/s)
        tar_change_steps_min = 100  # 目标变化最小步数
        tar_change_steps_max = 300  # 目标变化最大步数
        tar_dist_max = 3.0  # 目标最大距离 (m)
        goal_reach_threshold = 0.5  # 到达目标阈值 (m)

class H1ASELocationCfgPPO(LeggedRobotCfgPPO):
    class amp():
        amp_input_dim = H1ASELocationCfg.amp.num_obs_steps * H1ASELocationCfg.amp.num_obs_per_step
        amp_disc_hidden_dims = [1024, 512]
        amp_replay_buffer_size = 1000000
        amp_demo_buffer_size = 200000
        amp_demo_fetch_batch_size = 512
        amp_learn_batch_size = 4096
        amp_learning_rate = 1.e-4
        amp_reward_coef = 4.0
        amp_grad_pen = 5