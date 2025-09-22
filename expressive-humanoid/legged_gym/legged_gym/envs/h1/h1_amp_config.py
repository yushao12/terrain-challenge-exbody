from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1AMPCfg(LeggedRobotCfg):
    class motion:
        motion_curriculum = True
        motion_type = "yaml"
        motion_name = "motions_easywalk.yaml"  # 你可以修改这个文件名

        global_keybody = False
        global_keybody_reset_time = 2
        num_envs_as_motions = False
        no_keybody = False
        regen_pkl = False
        step_inplace_prob = 0.05
        resample_step_inplace_interval_s = 10

    class amp():
        num_obs_steps = 10
        num_obs_per_step = 19 + 3 + 3 + 3 + 12*3

class H1AMPCfgPPO(LeggedRobotCfgPPO):
    class amp():
        amp_input_dim = H1AMPCfg.amp.num_obs_steps * H1AMPCfg.amp.num_obs_per_step
        # 使用更深的discriminator架构
        amp_disc_hidden_dims = [1024, 1024, 512]  # 调整为三层网络架构
        amp_replay_buffer_size = 1000000
        amp_demo_buffer_size = 200000
        amp_demo_fetch_batch_size = 512
        amp_learn_batch_size = 4096
        # 学习率降低到当前的1/10
        amp_learning_rate = 1.e-5  # 从1.e-4调整为1.e-5
        amp_reward_coef = 4.0
       # amp_reward_coef = 0.0
        amp_grad_pen = 5 