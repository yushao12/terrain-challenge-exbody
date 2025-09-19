#!/usr/bin/env python3

"""
G1机器人视频录制脚本 - 使用虚拟显示录制G1机器人视频
"""

import sys
import os
import numpy as np
import time

# 添加legged_gym到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym'))

def record_g1_video():
    """录制G1机器人视频"""
    try:
        from legged_gym.envs import task_registry
        from legged_gym.utils import get_args
        
        print("正在创建G1机器人环境用于视频录制...")
        
        # 创建参数
        args = get_args()
        args.task = "g1"
        args.headless = False  # 需要显示来录制视频
        args.num_envs = 1
        args.seed = 42
        
        # 获取任务配置
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        
        # 配置用于视频录制
        env_cfg.env.num_envs = 1
        env_cfg.env.episode_length_s = 30  # 30秒的视频
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.mesh_type = 'trimesh'
        
        # 平坦地形
        env_cfg.terrain.terrain_kwargs = {
            'slope_treshold': 0.75,
            'difficulty': 0.0,  # 最简单的地形
            'downsampled_scale': 0.2,
            'pad': True,
            'discrete_obstacles': False,
            'random_uniform': False,
            'curriculum': False,
        }
        
        # 创建环境
        from legged_gym.envs import task_registry
        
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        
        print("✓ G1机器人环境创建成功!")
        print(f"✓ 机器人名称: {env_cfg.asset.name}")
        print(f"✓ URDF文件: {env_cfg.asset.file}")
        print(f"✓ 动作维度: {env_cfg.env.num_actions}")
        print(f"✓ 观测维度: {env_cfg.env.num_observations}")
        
        # 获取初始观测
        obs = env.get_observations()
        print(f"✓ 观测形状: {obs.shape}")
        
        # 录制视频
        print("\n开始录制G1机器人视频...")
        print("机器人将保持站立姿态并添加轻微动作")
        
        # 创建视频文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = f"g1_robot_demo_{timestamp}.mp4"
        
        # 设置视频录制
        env.gym.start_accessing_tensors(env.sim)
        
        # 运行录制
        actions = np.zeros((env.num_envs, env.num_actions))
        
        step_count = 0
        max_steps = 3000  # 约30秒的视频 (100 FPS)
        
        try:
            while step_count < max_steps:
                # 添加一些轻微的随机动作来展示机器人
                if step_count % 100 == 0:  # 每100步添加一些随机动作
                    actions = np.random.normal(0, 0.05, actions.shape)
                elif step_count % 200 == 0:  # 每200步重置为零动作
                    actions = np.zeros((env.num_envs, env.num_actions))
                
                # 执行动作
                obs, rewards, dones, infos = env.step(actions)
                step_count += 1
                
                # 每500步打印一次信息
                if step_count % 500 == 0:
                    print(f"录制步骤 {step_count}/{max_steps}: 奖励 = {rewards[0]:.3f}")
                
                # 重置完成的环境
                if dones.any():
                    env_ids = np.where(dones)[0]
                    env.reset_idx(env_ids)
                    obs = env.get_observations()
                    print(f"环境重置 (步骤 {step_count})")
                
        except KeyboardInterrupt:
            print(f"\n用户中断录制，总共录制了 {step_count} 步")
        
        env.gym.stop_accessing_tensors(env.sim)
        env.close()
        
        print(f"✓ 视频录制完成! 文件保存为: {video_filename}")
        print("您可以使用以下命令查看视频:")
        print(f"  ffplay {video_filename}")
        print(f"  vlc {video_filename}")
        
    except Exception as e:
        print(f"✗ 录制G1机器人视频时出现错误: {e}")
        import traceback
        traceback.print_exc()

def record_g1_terrain_video():
    """录制G1机器人在不同地形中的视频"""
    try:
        from legged_gym.envs import task_registry
        from legged_gym.utils import get_args
        
        print("正在创建G1机器人多地形环境用于视频录制...")
        
        # 创建参数
        args = get_args()
        args.task = "g1"
        args.headless = False
        args.num_envs = 4  # 4个环境展示不同地形
        args.seed = 42
        
        # 获取任务配置
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        
        # 配置用于多地形视频录制
        env_cfg.env.num_envs = 4
        env_cfg.env.episode_length_s = 60  # 60秒的视频
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 2
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.mesh_type = 'trimesh'
        
        # 不同难度的地形
        env_cfg.terrain.terrain_kwargs = {
            'slope_treshold': 0.75,
            'difficulty': 0.6,  # 中等难度
            'downsampled_scale': 0.2,
            'pad': True,
            'discrete_obstacles': True,  # 启用障碍物
            'random_uniform': True,
            'curriculum': False,
        }
        
        # 创建环境
        from legged_gym.envs import task_registry
        
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        
        print("✓ G1机器人多地形环境创建成功!")
        print(f"✓ 环境数量: {env_cfg.env.num_envs}")
        print("✓ 展示不同地形类型")
        
        # 录制视频
        print("\n开始录制G1机器人多地形视频...")
        
        # 创建视频文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = f"g1_robot_terrain_demo_{timestamp}.mp4"
        
        # 设置视频录制
        env.gym.start_accessing_tensors(env.sim)
        
        # 运行录制
        actions = np.zeros((env.num_envs, env.num_actions))
        
        step_count = 0
        max_steps = 6000  # 约60秒的视频
        
        try:
            while step_count < max_steps:
                # 添加一些随机动作
                if step_count % 200 == 0:
                    actions = np.random.normal(0, 0.03, actions.shape)
                elif step_count % 400 == 0:
                    actions = np.zeros((env.num_envs, env.num_actions))
                
                # 执行动作
                obs, rewards, dones, infos = env.step(actions)
                step_count += 1
                
                # 每1000步打印一次信息
                if step_count % 1000 == 0:
                    print(f"录制步骤 {step_count}/{max_steps}: 平均奖励 = {rewards.mean():.3f}")
                
                # 重置完成的环境
                if dones.any():
                    env_ids = np.where(dones)[0]
                    env.reset_idx(env_ids)
                    obs = env.get_observations()
                    print(f"环境重置 (步骤 {step_count})")
                
        except KeyboardInterrupt:
            print(f"\n用户中断录制，总共录制了 {step_count} 步")
        
        env.gym.stop_accessing_tensors(env.sim)
        env.close()
        
        print(f"✓ 多地形视频录制完成! 文件保存为: {video_filename}")
        print("您可以使用以下命令查看视频:")
        print(f"  ffplay {video_filename}")
        print(f"  vlc {video_filename}")
        
    except Exception as e:
        print(f"✗ 录制G1机器人多地形视频时出现错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("G1机器人视频录制工具")
    print("=" * 50)
    print("选择录制模式:")
    print("1. 基础录制 (平坦地形)")
    print("2. 多地形录制 (不同难度地形)")
    print("3. 退出")
    
    while True:
        try:
            choice = input("\n请选择 (1-3): ").strip()
            
            if choice == "1":
                print("\n启动基础录制...")
                record_g1_video()
                break
            elif choice == "2":
                print("\n启动多地形录制...")
                record_g1_terrain_video()
                break
            elif choice == "3":
                print("退出")
                break
            else:
                print("无效选择，请输入 1-3")
                
        except KeyboardInterrupt:
            print("\n用户中断")
            break
        except Exception as e:
            print(f"输入错误: {e}")

if __name__ == "__main__":
    main()