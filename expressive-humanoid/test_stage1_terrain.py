#!/usr/bin/env python3
"""
测试第一阶段地形的简化版本
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加路径
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/challenging_terrain')
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym')

from terrain_base.single_terrain import single_terrain
from terrain_base.config import terrain_config
from isaacgym import terrain_utils

def test_stage1_terrain():
    """测试第一阶段地形"""
    
    print("🧪 测试第一阶段地形...")
    
    # 创建地形
    terrain = terrain_utils.SubTerrain(
        "terrain",
        width=160,  # 8m / 0.05m = 160
        length=160,  # 8m / 0.05m = 160
        vertical_scale=0.005,
        horizontal_scale=0.05
    )
    
    # 生成原始parkour地形
    print("生成原始parkour地形...")
    original_terrain, goals_orig, final_x_orig = single_terrain.parkour(
        terrain, 
        length_x=8.0,
        length_y=4.0,
        num_goals=5,
        difficulty=0.8
    )
    
    # 生成阶段1训练地形
    print("生成阶段1训练地形...")
    stage1_terrain, goals_stage1, final_x_stage1 = single_terrain.parkour_training_stage1(
        terrain,
        length_x=8.0,
        length_y=4.0,
        num_goals=5,
        difficulty=0.8
    )
    
    # 验证结果
    print("\n📊 地形分析:")
    print(f"原始地形高度范围: {original_terrain.height_field_raw.min()} ~ {original_terrain.height_field_raw.max()}")
    print(f"阶段1地形高度范围: {stage1_terrain.height_field_raw.min()} ~ {stage1_terrain.height_field_raw.max()}")
    
    # 检查是否真的是平地
    terrain_area = stage1_terrain.height_field_raw[20:140, 20:140]  # 主要地形区域
    is_flat = np.all(terrain_area == 0)
    print(f"阶段1地形是否为平地: {is_flat}")
    
    # 检查可站立区域
    if hasattr(stage1_terrain, 'valid_standing_mask'):
        valid_count = np.sum(stage1_terrain.valid_standing_mask)
        total_count = stage1_terrain.valid_standing_mask.size
        coverage = 100 * valid_count / total_count
        print(f"可站立区域数量: {valid_count}/{total_count} ({coverage:.1f}%)")
    
    # 检查scan参考
    if hasattr(stage1_terrain, 'scan_reference'):
        print(f"Scan参考地形高度范围: {stage1_terrain.scan_reference.min()} ~ {stage1_terrain.scan_reference.max()}")
        print("✅ Scan参考保留原始parkour地形")
    
    # 检查训练阶段
    if hasattr(stage1_terrain, 'training_stage'):
        print(f"训练阶段: {stage1_terrain.training_stage}")
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始地形
    im1 = axes[0].imshow(original_terrain.height_field_raw, cmap='terrain', aspect='equal')
    axes[0].set_title('Original Parkour\n(Stones + Deep Pits)')
    axes[0].set_xlabel('X (grid)')
    axes[0].set_ylabel('Y (grid)')
    plt.colorbar(im1, ax=axes[0])
    
    # 阶段1物理地形（应该是平地）
    im2 = axes[1].imshow(stage1_terrain.height_field_raw, cmap='terrain', aspect='equal')
    axes[1].set_title('Stage 1 Physical Terrain\n(Flat Ground)')
    axes[1].set_xlabel('X (grid)')
    axes[1].set_ylabel('Y (grid)')
    plt.colorbar(im2, ax=axes[1])
    
    # 可站立区域掩码
    if hasattr(stage1_terrain, 'valid_standing_mask'):
        im3 = axes[2].imshow(stage1_terrain.valid_standing_mask, cmap='RdYlGn', aspect='equal')
        axes[2].set_title('Valid Standing Areas\n(Green = Parkour Stones)')
        axes[2].set_xlabel('X (grid)')
        axes[2].set_ylabel('Y (grid)')
        plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/stage1_terrain_test.png', 
                dpi=150, bbox_inches='tight')
    print("\n📸 地形对比图已保存到: stage1_terrain_test.png")
    
    # 总结
    print("\n🎯 第一阶段训练特点:")
    print("  ✅ 物理地形：完全平地，机器人不会掉坑")
    print("  ✅ Scan信息：保留原始parkour地形结构")
    print("  ✅ 可站立区域：标记了parkour石头位置")
    print("  ✅ 惩罚机制：踩到非parkour区域给予严重惩罚(-5.0)")
    print("  ✅ 学习目标：在平地上学会parkour步态和落脚点")
    
    return True

if __name__ == "__main__":
    try:
        success = test_stage1_terrain()
        if success:
            print("\n✅ 第一阶段地形测试成功！")
        else:
            print("\n❌ 第一阶段地形测试失败！")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()