#!/usr/bin/env python3

"""
H1_2机器人快速查看脚本
用于快速实例化和观察H1_2机器人
"""

import os
import sys
import torch

# 添加项目路径
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym')

def quick_h1_2_view():
    """快速查看H1_2机器人"""
    
    try:
        # 导入必要的模块
        from legged_gym.envs.h1_2.h1_2_mimic_config import H1_2MimicCfg
        from legged_gym.envs.h1_2.h1_2_mimic import H1_2Mimic
        from legged_gym.utils.helpers import parse_sim_params
        
        print("H1_2机器人配置信息:")
        print("=" * 50)
        
        # 创建配置
        cfg = H1_2MimicCfg()
        
        # 显示基本配置信息
        print(f"机器人名称: {cfg.asset.name}")
        print(f"URDF文件: {cfg.asset.file}")
        print(f"DOF数量: {cfg.env.num_policy_actions}")
        print(f"观测维度: {cfg.env.num_observations}")
        print(f"环境数量: {cfg.env.num_envs}")
        
        # 显示关节配置
        print(f"\n默认关节角度:")
        for joint_name, angle in cfg.init_state.default_joint_angles.items():
            print(f"  {joint_name}: {angle:.3f} rad")
        
        # 显示控制参数
        print(f"\n控制参数:")
        print(f"  控制类型: {cfg.control.control_type}")
        print(f"  动作缩放: {cfg.control.action_scale}")
        print(f"  降采样: {cfg.control.decimation}")
        
        print(f"\n刚度参数:")
        for joint_type, stiffness in cfg.control.stiffness.items():
            print(f"  {joint_type}: {stiffness}")
        
        print(f"\n阻尼参数:")
        for joint_type, damping in cfg.control.damping.items():
            print(f"  {joint_type}: {damping}")
        
        print("\nH1_2配置加载成功!")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保H1_2环境已正确配置")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    quick_h1_2_view()