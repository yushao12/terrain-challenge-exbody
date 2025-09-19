#!/usr/bin/env python3
"""
启动Parkour第一阶段训练的脚本
"""

import os
import sys
import subprocess

def start_stage1_training():
    """启动第一阶段训练"""
    
    print("🏃‍♂️ 启动Parkour第一阶段训练...")
    print("=" * 50)
    
    # 检查配置
    print("📋 检查配置...")
    
    # 检查地形配置
    combine_config_path = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/challenging_terrain/terrain_base/combine_config.py"
    with open(combine_config_path, 'r') as f:
        content = f.read()
        if '("single", 1, 1)' in content:
            print("✅ 地形配置：使用parkour_training_stage1")
        else:
            print("❌ 地形配置：未找到parkour_training_stage1配置")
            return False
    
    # 检查训练配置
    config_path = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/legged_gym/envs/h1/h1_mimic_config.py"
    with open(config_path, 'r') as f:
        content = f.read()
        if 'training_stage = 1' in content and 'two_stage_training = True' in content:
            print("✅ 训练配置：第一阶段训练已启用")
        else:
            print("❌ 训练配置：第一阶段训练未正确配置")
            return False
    
    print("\n🎯 第一阶段训练特点：")
    print("  - 物理地形：平地，便于学习基本步态")
    print("  - Scan信息：原始parkour地形")
    print("  - 奖励：脚部接触惩罚，引导学习正确落脚点")
    print("  - 目标：学会基本的parkour步态模式")
    
    print("\n🚀 启动训练命令：")
    train_cmd = [
        "python", "train.py",
        "--task", "h1_mimic",
        "--num_envs", "4096",
        "--headless"
    ]
    
    print(" ".join(train_cmd))
    
    # 切换到训练目录
    train_dir = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/legged_gym/scripts"
    
    print(f"\n📁 切换到训练目录: {train_dir}")
    os.chdir(train_dir)
    
    # 询问是否启动训练
    response = input("\n❓ 是否启动训练？(y/n): ")
    if response.lower() == 'y':
        print("\n🏃‍♂️ 启动训练...")
        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 训练启动失败: {e}")
            return False
        except KeyboardInterrupt:
            print("\n⏹️ 训练被用户中断")
            return True
    else:
        print("⏸️ 训练未启动")
    
    return True

if __name__ == "__main__":
    start_stage1_training()