from legged_gym.envs.h1.h1_mimic_config import H1MimicCfg, H1MimicCfgPPO

class H1MimicAMPCfg( H1MimicCfg ):
    class motion:
        motion_curriculum = True
        motion_type = "yaml"
        motion_name = "motions_easywalk.yaml"  # 使用你的easy_walking motion

        global_keybody = False
        global_keybody_reset_time = 2

        num_envs_as_motions = False

        no_keybody = False
        regen_pkl = False

        step_inplace_prob = 0.05
        resample_step_inplace_interval_s = 10

    class amp():
        num_obs_steps = 10
        # 简化观测：dof_pos(19) + local_root_vel(3) + local_root_ang_vel(3) + roll(1) + pitch(1) + root_height(1) + lower_body_key_pos(6*3)
        num_obs_per_step = 19 + 3 + 3 + 1 + 1 + 1 + 6*3

class H1MimicAMPCfgPPO( H1MimicCfgPPO ):
    class runner( H1MimicCfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimicAMP"
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
    
    class amp():
        amp_input_dim = H1MimicAMPCfg.amp.num_obs_steps * H1MimicAMPCfg.amp.num_obs_per_step
        amp_disc_hidden_dims = [1024, 512]

        amp_replay_buffer_size = 1000000
        amp_demo_buffer_size = 200000
        amp_demo_fetch_batch_size = 512
        amp_learn_batch_size = 4096
        amp_learning_rate = 1.e-4

        amp_reward_coef = 4.0

        amp_grad_pen = 5