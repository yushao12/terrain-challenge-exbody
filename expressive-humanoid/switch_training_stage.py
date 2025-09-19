#!/usr/bin/env python3
"""
切换Parkour训练阶段的脚本
"""

import os
import sys

def switch_training_stage(stage):
    """切换训练阶段"""
    
    if stage not in [1, 2]:
        print("❌ 错误：阶段必须是1或2")
        return False
    
    config_path = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/legged_gym/envs/h1/h1_mimic_config.py"
    
    print(f"🔄 切换到训练阶段 {stage}...")
    
    # 读取配置文件
    with open(config_path, 'r') as f:
        content = f.read()
    
    # 更新训练阶段
    old_line = f"training_stage = {3-stage}"  # 1->2, 2->1
    new_line = f"training_stage = {stage}"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        print(f"✅ 更新训练阶段: {old_line} -> {new_line}")
    else:
        print(f"⚠️ 未找到 {old_line}，可能已经是阶段 {stage}")
    
    # 更新地形配置
    combine_config_path = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/challenging_terrain/terrain_base/combine_config.py"
    
    with open(combine_config_path, 'r') as f:
        terrain_content = f.read()
    
    if stage == 1:
        # 阶段1：使用parkour_training_stage1
        if '("single", 0, 1)' in terrain_content:  # 原始parkour
            terrain_content = terrain_content.replace('("single", 0, 1)', '("single", 1, 1)')
            print("✅ 地形配置：切换到parkour_training_stage1")
        elif '("single", 1, 1)' in terrain_content:
            print("✅ 地形配置：已经是parkour_training_stage1")
    else:
        # 阶段2：使用原始parkour
        if '("single", 1, 1)' in terrain_content:  # parkour_training_stage1
            terrain_content = terrain_content.replace('("single", 1, 1)', '("single", 0, 1)')
            print("✅ 地形配置：切换到原始parkour")
        elif '("single", 0, 1)' in terrain_content:
            print("✅ 地形配置：已经是原始parkour")
    
    # 写回配置文件
    with open(config_path, 'w') as f:
        f.write(content)
    
    with open(combine_config_path, 'w') as f:
        f.write(terrain_content)
    
    print(f"\n🎯 阶段 {stage} 训练特点：")
    if stage == 1:
        print("  - 物理地形：平地，便于学习基本步态")
        print("  - Scan信息：原始parkour地形")
        print("  - 奖励：脚部接触惩罚，引导学习正确落脚点")
        print("  - 目标：学会基本的parkour步态模式")
    else:
        print("  - 物理地形：真实parkour地形")
        print("  - Scan信息：原始parkour地形")
        print("  - 奖励：正常奖励系统")
        print("  - 目标：在真实地形上微调和最终训练")
    
    return True

def show_current_stage():
    """显示当前训练阶段"""
    
    config_path = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/legged_gym/envs/h1/h1_mimic_config.py"
    combine_config_path = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/challenging_terrain/terrain_base/combine_config.py"
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    with open(combine_config_path, 'r') as f:
        terrain_content = f.read()
    
    # 提取训练阶段
    for line in content.split('\n'):
        if 'training_stage =' in line:
            stage = line.split('=')[1].strip()
            print(f"📊 当前训练阶段: {stage}")
            break
    
    # 检查地形配置
    if '("single", 1, 1)' in terrain_content:
        print("🏔️ 当前地形: parkour_training_stage1 (阶段1)")
    elif '("single", 0, 1)' in terrain_content:
        print("🏔️ 当前地形: 原始parkour (阶段2)")
    else:
        print("🏔️ 当前地形: 其他地形")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "show":
            show_current_stage()
        else:
            try:
                stage = int(sys.argv[1])
                switch_training_stage(stage)
            except ValueError:
                print("❌ 错误：阶段必须是数字")
    else:
        print("使用方法:")
        print("  python switch_training_stage.py 1    # 切换到阶段1")
        print("  python switch_training_stage.py 2    # 切换到阶段2")
        print("  python switch_training_stage.py show # 显示当前阶段")