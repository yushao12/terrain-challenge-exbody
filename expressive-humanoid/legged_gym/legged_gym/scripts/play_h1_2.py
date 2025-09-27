#!/usr/bin/env python3

"""
H1_2机器人play脚本
用于播放H1_2策略或可视化motion数据
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym')

def play_h1_2_policy(policy_path=None, task_name="h1_2_mimic_amp"):
    """播放H1_2策略"""
    
    try:
        from legged_gym.utils.task_registry import task_registry
        from legged_gym.utils.helpers import get_args, export_policy_as_jit
        
        print(f"Playing H1_2 task: {task_name}")
        
        # 获取任务配置
        env_cfg, train_cfg = task_registry.get_cfgs(task_name)
        
        # 创建环境
        env, _ = task_registry.make_env(name=task_name, args=None)
        
        print(f"Environment created successfully!")
        print(f"Number of environments: {env.num_envs}")
        print(f"Observation space: {env.num_obs}")
        print(f"Action space: {env.num_actions}")
        
        # 如果没有提供策略路径，使用随机动作
        if policy_path is None:
            print("No policy provided, using random actions")
            
            # 运行几步随机动作
            for i in range(100):
                actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
                obs, rewards, dones, infos = env.step(actions)
                
                if i % 10 == 0:
                    print(f"Step {i}: Reward = {rewards.mean().item():.3f}")
        
        else:
            print(f"Loading policy from: {policy_path}")
            # 这里可以加载训练好的策略
            # policy = torch.jit.load(policy_path)
            # 然后使用策略进行推理
        
        print("H1_2 play completed!")
        
    except Exception as e:
        print(f"Error playing H1_2: {e}")
        import traceback
        traceback.print_exc()

def visualize_h1_2_motion(motion_file=None):
    """可视化H1_2的motion数据"""
    
    try:
        from legged_gym.utils.task_registry import task_registry
        
        print("Visualizing H1_2 motion data...")
        
        # 使用h1_2_view任务来可视化motion
        task_name = "h1_2_view"
        
        # 获取任务配置
        env_cfg, train_cfg = task_registry.get_cfgs(task_name)
        
        # 如果有指定的motion文件，更新配置
        if motion_file:
            env_cfg.motion.motion_name = motion_file
            print(f"Using motion file: {motion_file}")
        
        # 创建环境
        env, _ = task_registry.make_env(name=task_name, args=None)
        
        print(f"H1_2 motion visualization environment created!")
        print(f"Number of environments: {env.num_envs}")
        
        # 运行可视化
        for i in range(1000):
            # 让环境自动播放motion数据
            obs, rewards, dones, infos = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
            
            if i % 100 == 0:
                print(f"Visualization step {i}")
        
        print("H1_2 motion visualization completed!")
        
    except Exception as e:
        print(f"Error visualizing H1_2 motion: {e}")
        import traceback
        traceback.print_exc()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Play H1_2 robot')
    parser.add_argument('--task', type=str, default='h1_2_mimic_amp', 
                       help='Task name to play')
    parser.add_argument('--policy', type=str, default=None,
                       help='Path to policy file')
    parser.add_argument('--motion', type=str, default=None,
                       help='Motion file to visualize')
    parser.add_argument('--mode', type=str, choices=['policy', 'motion'], default='policy',
                       help='Play mode: policy or motion')
    
    args = parser.parse_args()
    
    if args.mode == 'policy':
        play_h1_2_policy(args.policy, args.task)
    elif args.mode == 'motion':
        visualize_h1_2_motion(args.motion)

if __name__ == "__main__":
    main()