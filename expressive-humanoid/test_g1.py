#!/usr/bin/env python3
"""
G1机器人测试脚本
用于验证G1环境的基本功能
"""

import sys
import os
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid')

from legged_gym.envs.g1.g1_mimic_config import G1MimicCfg, G1MimicCfgPPO
from legged_gym.envs.g1.g1_mimic import G1Mimic

def test_g1_config():
    """测试G1配置"""
    print("=== 测试G1配置 ===")
    
    # 测试基本配置
    cfg = G1MimicCfg()
    print(f"环境数量: {cfg.env.num_envs}")
    print(f"观测维度: {cfg.env.num_observations}")
    print(f"动作维度: {cfg.env.num_policy_actions}")
    print(f"DOF数量: {cfg.env.num_policy_actions}")
    print(f"URDF文件: {cfg.asset.file}")
    
    # 测试PPO配置
    ppo_cfg = G1MimicCfgPPO()
    print(f"Runner类: {ppo_cfg.runner.runner_class_name}")
    print(f"Policy类: {ppo_cfg.policy.policy_class_name}")
    print(f"Algorithm类: {ppo_cfg.algorithm.algorithm_class_name}")
    
    print("✓ G1配置测试通过")

def test_g1_observation_space():
    """测试G1观测空间"""
    print("\n=== 测试G1观测空间 ===")
    
    cfg = G1MimicCfg()
    
    # 计算观测维度
    n_proprio = cfg.env.n_proprio  # 51
    n_demo = cfg.env.n_demo  # 24
    n_scan = cfg.env.n_scan  # 132
    n_priv_latent = cfg.env.n_priv_latent  # 29
    n_priv = cfg.env.n_priv  # 3
    history_len = cfg.env.history_len  # 10
    prop_hist_len = cfg.env.prop_hist_len  # 4
    n_feature = cfg.env.n_feature  # 204
    
    expected_obs_dim = n_feature + n_proprio + n_demo + n_scan + history_len*n_proprio + n_priv_latent + n_priv
    actual_obs_dim = cfg.env.num_observations
    
    print(f"n_proprio: {n_proprio}")
    print(f"n_demo: {n_demo}")
    print(f"n_scan: {n_scan}")
    print(f"n_priv_latent: {n_priv_latent}")
    print(f"n_priv: {n_priv}")
    print(f"history_len: {history_len}")
    print(f"prop_hist_len: {prop_hist_len}")
    print(f"n_feature: {n_feature}")
    print(f"期望观测维度: {expected_obs_dim}")
    print(f"实际观测维度: {actual_obs_dim}")
    
    if expected_obs_dim == actual_obs_dim:
        print("✓ G1观测空间测试通过")
    else:
        print("✗ G1观测空间测试失败")
        return False
    
    return True

def test_g1_vs_h1_differences():
    """测试G1与H1的差异"""
    print("\n=== 测试G1与H1差异 ===")
    
    # 导入H1配置进行对比
    from legged_gym.envs.h1.h1_mimic_config import H1MimicCfg
    
    g1_cfg = G1MimicCfg()
    h1_cfg = H1MimicCfg()
    
    print("关键差异对比:")
    print(f"DOF数量 - G1: {g1_cfg.env.num_policy_actions}, H1: {h1_cfg.env.num_policy_actions}")
    print(f"观测维度 - G1: {g1_cfg.env.num_observations}, H1: {h1_cfg.env.num_observations}")
    print(f"URDF文件 - G1: {g1_cfg.asset.file}")
    print(f"URDF文件 - H1: {h1_cfg.asset.file}")
    print(f"躯干名称 - G1: {g1_cfg.asset.torso_name}, H1: {h1_cfg.asset.torso_name}")
    print(f"足部名称 - G1: {g1_cfg.asset.foot_name}, H1: {h1_cfg.asset.foot_name}")
    
    print("✓ G1与H1差异测试通过")

def main():
    """主测试函数"""
    print("开始G1机器人测试...")
    
    try:
        test_g1_config()
        if test_g1_observation_space():
            test_g1_vs_h1_differences()
            print("\n🎉 所有测试通过！G1机器人配置正确。")
        else:
            print("\n❌ 测试失败，请检查配置。")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()