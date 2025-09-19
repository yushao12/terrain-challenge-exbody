#!/usr/bin/env python3

"""
测试G1机器人加载脚本
"""

import sys
import os

# 添加legged_gym到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym'))

def test_g1_loading():
    """测试G1机器人是否能正常加载"""
    try:
        print("测试G1机器人加载...")
        
        # 测试导入
        from legged_gym.envs import task_registry
        print("✓ 成功导入task_registry")
        
        # 测试获取G1配置
        env_cfg, train_cfg = task_registry.get_cfgs(name="g1")
        print("✓ 成功获取G1配置")
        
        # 检查配置
        print(f"✓ 机器人名称: {env_cfg.asset.name}")
        print(f"✓ URDF文件: {env_cfg.asset.file}")
        print(f"✓ 动作维度: {env_cfg.env.num_actions}")
        print(f"✓ 观测维度: {env_cfg.env.num_observations}")
        
        # 检查URDF文件是否存在
        urdf_path = env_cfg.asset.file.replace('{LEGGED_GYM_ROOT_DIR}', '/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym')
        if os.path.exists(urdf_path):
            print(f"✓ URDF文件存在: {urdf_path}")
        else:
            print(f"✗ URDF文件不存在: {urdf_path}")
            return False
        
        # 测试创建环境（不运行）
        print("\n测试环境创建...")
        from legged_gym.utils import get_args
        
        args = get_args()
        args.task = "g1"
        args.headless = True  # 无头模式，不显示GUI
        args.num_envs = 1
        args.seed = 1
        
        # 简化配置
        env_cfg.env.num_envs = 1
        env_cfg.env.episode_length_s = 10
        
        # 创建环境
        from legged_gym.envs import task_registry
        
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        print("✓ 成功创建G1环境")
        
        # 测试获取观测
        obs = env.get_observations()
        print(f"✓ 成功获取观测，形状: {obs.shape}")
        
        # 测试执行一步
        actions = env.get_actions()
        obs, rewards, dones, infos = env.step(actions)
        print("✓ 成功执行一步")
        
        env.close()
        print("✓ 成功关闭环境")
        
        print("\n🎉 G1机器人加载测试全部通过!")
        print("现在您可以运行可视化脚本查看G1机器人:")
        print("  python quick_g1_view.py")
        print("  python visualize_g1.py")
        
        return True
        
    except Exception as e:
        print(f"✗ G1机器人加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_g1_loading()
    sys.exit(0 if success else 1)