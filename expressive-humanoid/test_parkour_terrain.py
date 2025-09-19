#!/usr/bin/env python3
"""
测试新的parkour训练阶段1地形变体
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

def test_parkour_terrain_variant():
    """测试新的parkour地形变体"""
    
    print("测试parkour训练阶段1地形变体...")
    
    # 创建地形
    terrain = terrain_utils.SubTerrain(
        "terrain",
        width=160,  # 8m / 0.05m = 160
        length=160,  # 8m / 0.05m = 160
        vertical_scale=0.005,
        horizontal_scale=0.05
    )
    
    # 测试原始parkour地形
    print("生成原始parkour地形...")
    original_terrain, goals_orig, final_x_orig = single_terrain.parkour(
        terrain, 
        length_x=8.0,
        length_y=4.0,
        num_goals=5,
        difficulty=0.8
    )
    
    # 测试新的parkour训练阶段1地形
    print("生成parkour训练阶段1地形...")
    stage1_terrain, goals_stage1, final_x_stage1 = single_terrain.parkour_training_stage1(
        terrain,
        length_x=8.0,
        length_y=4.0,
        num_goals=5,
        difficulty=0.8
    )
    
    # 验证结果
    print(f"原始地形形状: {original_terrain.height_field_raw.shape}")
    print(f"阶段1地形形状: {stage1_terrain.height_field_raw.shape}")
    print(f"原始地形高度范围: {original_terrain.height_field_raw.min()} ~ {original_terrain.height_field_raw.max()}")
    print(f"阶段1地形高度范围: {stage1_terrain.height_field_raw.min()} ~ {stage1_terrain.height_field_raw.max()}")
    
    # 检查是否有scan参考和有效站立掩码
    if hasattr(stage1_terrain, 'scan_reference'):
        print(f"Scan参考地形形状: {stage1_terrain.scan_reference.shape}")
        print(f"Scan参考地形高度范围: {stage1_terrain.scan_reference.min()} ~ {stage1_terrain.scan_reference.max()}")
    else:
        print("警告: 没有找到scan_reference")
    
    if hasattr(stage1_terrain, 'valid_standing_mask'):
        print(f"有效站立掩码形状: {stage1_terrain.valid_standing_mask.shape}")
        print(f"可站立区域数量: {np.sum(stage1_terrain.valid_standing_mask)}")
    else:
        print("警告: 没有找到valid_standing_mask")
    
    if hasattr(stage1_terrain, 'training_stage'):
        print(f"训练阶段: {stage1_terrain.training_stage}")
    else:
        print("警告: 没有找到training_stage")
    
    # 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始地形
    im1 = axes[0, 0].imshow(original_terrain.height_field_raw, cmap='terrain', aspect='equal')
    axes[0, 0].set_title('原始Parkour地形')
    axes[0, 0].set_xlabel('X (grid)')
    axes[0, 0].set_ylabel('Y (grid)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 阶段1物理地形
    im2 = axes[0, 1].imshow(stage1_terrain.height_field_raw, cmap='terrain', aspect='equal')
    axes[0, 1].set_title('阶段1物理地形（平地）')
    axes[0, 1].set_xlabel('X (grid)')
    axes[0, 1].set_ylabel('Y (grid)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Scan参考地形
    if hasattr(stage1_terrain, 'scan_reference'):
        im3 = axes[1, 0].imshow(stage1_terrain.scan_reference, cmap='terrain', aspect='equal')
        axes[1, 0].set_title('Scan参考地形（原始Parkour）')
        axes[1, 0].set_xlabel('X (grid)')
        axes[1, 0].set_ylabel('Y (grid)')
        plt.colorbar(im3, ax=axes[1, 0])
    
    # 有效站立掩码
    if hasattr(stage1_terrain, 'valid_standing_mask'):
        im4 = axes[1, 1].imshow(stage1_terrain.valid_standing_mask, cmap='RdYlGn', aspect='equal')
        axes[1, 1].set_title('有效站立区域掩码')
        axes[1, 1].set_xlabel('X (grid)')
        axes[1, 1].set_ylabel('Y (grid)')
        plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/parkour_terrain_comparison.png', dpi=150, bbox_inches='tight')
    print("地形对比图已保存到: parkour_terrain_comparison.png")
    
    return True

if __name__ == "__main__":
    try:
        success = test_parkour_terrain_variant()
        if success:
            print("✅ 测试成功！新的parkour地形变体工作正常。")
        else:
            print("❌ 测试失败！")
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()