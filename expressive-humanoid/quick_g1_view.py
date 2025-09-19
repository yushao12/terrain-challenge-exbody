#!/usr/bin/env python3

"""
快速G1机器人查看脚本
"""

import sys
import os
import numpy as np

# 添加legged_gym到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym'))

def quick_g1_view():
    """快速查看G1机器人"""
    try:
        from legged_gym.envs import task_registry
        from legged_gym.utils import get_args
        
        print("正在创建G1机器人环境...")
        
        # 创建参数
        args = get_args()
        args.task = "g1"
        args.headless = False
        args.num_envs = 1
        args.seed = 42
        
        # 获取任务配置
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        
        # 简化配置用于快速查看
        env_cfg.env.num_envs = 1
        env_cfg.env.episode_length_s = 50
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
        from legged_gym.envs import *
        from legged_gym.utils.task_registry import task_registry
        
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        
        print("✓ G1机器人环境创建成功!")
        print(f"✓ 机器人名称: {env_cfg.asset.name}")
        print(f"✓ URDF文件: {env_cfg.asset.file}")
        print(f"✓ 动作维度: {env_cfg.env.num_actions}")
        print(f"✓ 观测维度: {env_cfg.env.num_observations}")
        
        # 获取初始观测
        obs = env.get_observations()
        print(f"✓ 观测形状: {obs.shape}")
        
        # 运行几步来展示机器人
        print("\n开始展示G1机器人...")
        print("机器人将保持站立姿态")
        print("按 Ctrl+C 退出")
        
        # 零动作（保持默认站立姿态）
        actions = np.zeros((env.num_envs, env.num_actions))
        
        step_count = 0
        try:
            while True:
                # 执行动作
                obs, rewards, dones, infos = env.step(actions)
                step_count += 1
                
                # 每100步打印一次信息
                if step_count % 100 == 0:
                    print(f"步骤 {step_count}: 奖励 = {rewards[0]:.3f}")
                
                # 重置完成的环境
                if dones.any():
                    env_ids = np.where(dones)[0]
                    env.reset_idx(env_ids)
                    obs = env.get_observations()
                    print(f"环境重置 (步骤 {step_count})")
                
        except KeyboardInterrupt:
            print(f"\n用户中断，总共运行了 {step_count} 步")
        
        env.close()
        print("✓ 可视化完成!")
        
    except Exception as e:
        print(f"✗ 创建G1机器人环境时出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_g1_view()