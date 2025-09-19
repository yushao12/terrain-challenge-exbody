#!/usr/bin/env python3

"""
简单的G1机器人测试脚本
"""

import sys
import os

# 添加legged_gym到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym'))

def simple_g1_test():
    """简单的G1机器人测试"""
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
        
        print("\n🎉 G1机器人配置测试通过!")
        print("现在您可以运行以下命令来录制G1机器人视频:")
        print("  ./record_g1_xvfb.sh")
        print("  python record_g1_video.py")
        
        return True
        
    except Exception as e:
        print(f"✗ G1机器人测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_g1_test()
    sys.exit(0 if success else 1)