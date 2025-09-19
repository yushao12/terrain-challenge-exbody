#!/usr/bin/env python3

"""
G1机器人可视化脚本 - 在环境中显示G1机器人
"""

import sys
import os
import numpy as np

# 添加legged_gym到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym'))

def create_g1_visualization():
    """创建G1机器人可视化"""
    try:
        from legged_gym.envs import task_registry
        from legged_gym.utils import get_args, export_policy
        
        # 创建参数
        args = get_args()
        args.task = "g1"  # 使用基础G1任务
        args.headless = False  # 启用可视化
        args.num_envs = 1  # 只创建一个环境用于可视化
        args.seed = 1
        
        # 获取任务配置
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        
        # 修改配置以适合可视化
        env_cfg.env.num_envs = 1
        env_cfg.env.episode_length_s = 100  # 长时间运行以便观察
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False  # 不使用课程学习
        env_cfg.terrain.mesh_type = 'trimesh'  # 使用三角网格地形
        
        # 设置地形类型为平坦地形
        env_cfg.terrain.terrain_kwargs = {
            'slope_treshold': 0.75,
            'difficulty': 0.5,
            'downsampled_scale': 0.2,
            'pad': True,
            'discrete_obstacles': False,
            'random_uniform': False,
            'curriculum': False,
        }
        
        # 创建环境
        from legged_gym.envs import *
        from legged_gym.utils.task_registry import task_registry
        
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        
        print("✓ G1机器人环境创建成功!")
        print(f"✓ 机器人名称: {env_cfg.asset.name}")
        print(f"✓ URDF文件: {env_cfg.asset.file}")
        print(f"✓ 环境数量: {env_cfg.env.num_envs}")
        print(f"✓ 动作维度: {env_cfg.env.num_actions}")
        print(f"✓ 观测维度: {env_cfg.env.num_observations}")
        
        # 运行可视化
        print("\n开始G1机器人可视化...")
        print("按 Ctrl+C 退出可视化")
        
        obs = env.get_observations()
        
        # 简单的站立动作（所有关节保持默认角度）
        default_actions = np.zeros((env.num_envs, env.num_actions))
        
        for i in range(10000):  # 运行10000步
            # 使用默认动作（站立姿态）
            actions = default_actions.copy()
            
            # 可选：添加一些轻微的随机动作来观察机器人
            if i % 100 == 0:  # 每100步添加一些随机动作
                actions += np.random.normal(0, 0.1, actions.shape)
            
            # 执行动作
            obs, rewards, dones, infos = env.step(actions)
            
            # 重置完成的环境
            if dones.any():
                env_ids = np.where(dones)[0]
                env.reset_idx(env_ids)
                obs = env.get_observations()
            
            # 每1000步打印一次信息
            if i % 1000 == 0:
                print(f"步骤 {i}: 奖励 = {rewards[0]:.3f}")
        
        env.close()
        print("✓ 可视化完成!")
        
    except KeyboardInterrupt:
        print("\n用户中断可视化")
        if 'env' in locals():
            env.close()
    except Exception as e:
        print(f"✗ 可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def create_g1_terrain_visualization():
    """创建G1机器人在不同地形中的可视化"""
    try:
        from legged_gym.envs import task_registry
        from legged_gym.utils import get_args
        
        # 创建参数
        args = get_args()
        args.task = "g1"
        args.headless = False
        args.num_envs = 4  # 创建4个环境展示不同地形
        args.seed = 1
        
        # 获取任务配置
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        
        # 修改配置
        env_cfg.env.num_envs = 4
        env_cfg.env.episode_length_s = 200
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 2
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.mesh_type = 'trimesh'
        
        # 设置不同难度的地形
        env_cfg.terrain.terrain_kwargs = {
            'slope_treshold': 0.75,
            'difficulty': 0.8,  # 较高难度
            'downsampled_scale': 0.2,
            'pad': True,
            'discrete_obstacles': True,  # 启用障碍物
            'random_uniform': True,
            'curriculum': False,
        }
        
        # 创建环境
        from legged_gym.envs import *
        from legged_gym.utils.task_registry import task_registry
        
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        
        print("✓ G1机器人多地形环境创建成功!")
        print(f"✓ 环境数量: {env_cfg.env.num_envs}")
        print("✓ 展示不同地形类型")
        
        # 运行可视化
        print("\n开始G1机器人多地形可视化...")
        print("按 Ctrl+C 退出可视化")
        
        obs = env.get_observations()
        default_actions = np.zeros((env.num_envs, env.num_actions))
        
        for i in range(20000):
            actions = default_actions.copy()
            
            # 添加一些随机动作
            if i % 200 == 0:
                actions += np.random.normal(0, 0.05, actions.shape)
            
            obs, rewards, dones, infos = env.step(actions)
            
            if dones.any():
                env_ids = np.where(dones)[0]
                env.reset_idx(env_ids)
                obs = env.get_observations()
            
            if i % 2000 == 0:
                print(f"步骤 {i}: 平均奖励 = {rewards.mean():.3f}")
        
        env.close()
        print("✓ 多地形可视化完成!")
        
    except KeyboardInterrupt:
        print("\n用户中断可视化")
        if 'env' in locals():
            env.close()
    except Exception as e:
        print(f"✗ 多地形可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("G1机器人可视化工具")
    print("=" * 50)
    print("选择可视化模式:")
    print("1. 基础可视化 (平坦地形)")
    print("2. 多地形可视化 (不同难度地形)")
    print("3. 退出")
    
    while True:
        try:
            choice = input("\n请选择 (1-3): ").strip()
            
            if choice == "1":
                print("\n启动基础可视化...")
                create_g1_visualization()
                break
            elif choice == "2":
                print("\n启动多地形可视化...")
                create_g1_terrain_visualization()
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