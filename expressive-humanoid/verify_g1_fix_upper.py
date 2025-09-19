#!/usr/bin/env python3

"""
验证G1 fix upper配置的脚本
"""

import sys
import os

def verify_g1_fix_upper_config():
    """验证G1 fix upper的配置"""
    try:
        # 检查配置文件中的机器人名称
        config_file = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/legged_gym/envs/g1/g1_config.py"
        
        with open(config_file, 'r') as f:
            content = f.read()
            
        if 'name = "g1_fix_upper"' in content:
            print("✓ G1配置文件中机器人名称正确设置为 'g1_fix_upper'")
        else:
            print("✗ G1配置文件中机器人名称设置错误")
            return False
            
        if 'experiment_name = \'g1_fix\'' in content:
            print("✓ G1配置文件中实验名称正确设置为 'g1_fix'")
        else:
            print("✗ G1配置文件中实验名称设置错误")
            return False
            
        # 检查mimic配置文件
        mimic_config_file = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/legged_gym/envs/g1/g1_mimic_config.py"
        
        with open(mimic_config_file, 'r') as f:
            mimic_content = f.read()
            
        if 'name = "g1_fix_upper"' in mimic_content:
            print("✓ G1 mimic配置文件中机器人名称正确设置为 'g1_fix_upper'")
        else:
            print("✗ G1 mimic配置文件中机器人名称设置错误")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ 验证过程中出现错误: {e}")
        return False

def verify_urdf_file():
    """验证URDF文件是否存在"""
    try:
        urdf_path = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/resources/robots/g1/g1_12dof_with_hand.urdf"
        
        if os.path.exists(urdf_path):
            print("✓ G1 URDF文件存在")
            return True
        else:
            print(f"✗ G1 URDF文件不存在: {urdf_path}")
            return False
            
    except Exception as e:
        print(f"✗ 验证URDF文件时出现错误: {e}")
        return False

def main():
    """主函数"""
    print("验证G1 fix upper配置...")
    print("=" * 50)
    
    tests = [
        verify_g1_fix_upper_config,
        verify_urdf_file,
    ]
    
    results = []
    for test in tests:
        print(f"\n运行 {test.__name__}...")
        result = test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("验证结果:")
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 G1 fix upper配置验证成功!")
        print("\n现在您可以使用以下命令训练G1 fix upper:")
        print("  python legged_gym/scripts/train.py --task=g1")
        print("  python legged_gym/scripts/train.py --task=g1_mimic")
        print("  python legged_gym/scripts/train.py --task=g1_mimic_amp")
    else:
        print("❌ 部分验证失败，请检查上述问题。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)