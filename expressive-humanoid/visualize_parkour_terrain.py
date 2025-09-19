#!/usr/bin/env python3
"""
可视化Parkour地形划分和阶段1训练地形
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加路径
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/challenging_terrain')
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym')

from terrain_base.single_terrain import single_terrain
from terrain_base.config import terrain_config
from isaacgym import terrain_utils

def visualize_parkour_terrain():
    """可视化parkour地形的划分"""
    
    print("🎨 可视化Parkour地形划分...")
    
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
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 原始地形
    im1 = axes[0, 0].imshow(original_terrain.height_field_raw, cmap='terrain', aspect='equal')
    axes[0, 0].set_title('Original Parkour Terrain', fontsize=14)
    axes[0, 0].set_xlabel('X (grid)')
    axes[0, 0].set_ylabel('Y (grid)')
    plt.colorbar(im1, ax=axes[0, 0], label='Height (grid units)')
    
    # 阶段1物理地形
    im2 = axes[0, 1].imshow(stage1_terrain.height_field_raw, cmap='terrain', aspect='equal')
    axes[0, 1].set_title('Stage 1 Physical Terrain\n(Shallow Pits)', fontsize=14)
    axes[0, 1].set_xlabel('X (grid)')
    axes[0, 1].set_ylabel('Y (grid)')
    plt.colorbar(im2, ax=axes[0, 1], label='Height (grid units)')
    
    # 有效站立区域掩码
    im3 = axes[0, 2].imshow(stage1_terrain.valid_standing_mask, cmap='RdYlGn', aspect='equal')
    axes[0, 2].set_title('Valid Standing Areas\n(Green = Safe)', fontsize=14)
    axes[0, 2].set_xlabel('X (grid)')
    axes[0, 2].set_ylabel('Y (grid)')
    plt.colorbar(im3, ax=axes[0, 2], label='Valid (1) / Invalid (0)')
    
    # 地形高度对比
    height_diff = stage1_terrain.height_field_raw - original_terrain.height_field_raw
    im4 = axes[1, 0].imshow(height_diff, cmap='RdBu_r', aspect='equal')
    axes[1, 0].set_title('Height Difference\n(Stage1 - Original)', fontsize=14)
    axes[1, 0].set_xlabel('X (grid)')
    axes[1, 0].set_ylabel('Y (grid)')
    plt.colorbar(im4, ax=axes[1, 0], label='Height Difference (grid units)')
    
    # 叠加显示：站立区域 + 原始地形
    overlay = axes[1, 1].imshow(original_terrain.height_field_raw, cmap='terrain', aspect='equal', alpha=0.7)
    valid_areas = np.ma.masked_where(stage1_terrain.valid_standing_mask == 0, stage1_terrain.valid_standing_mask)
    axes[1, 1].imshow(valid_areas, cmap='RdYlGn', aspect='equal', alpha=0.5)
    axes[1, 1].set_title('Overlay: Original Terrain + Valid Areas', fontsize=14)
    axes[1, 1].set_xlabel('X (grid)')
    axes[1, 1].set_ylabel('Y (grid)')
    plt.colorbar(overlay, ax=axes[1, 1], label='Height (grid units)')
    
    # 地形特征统计
    axes[1, 2].axis('off')
    stats_text = f"""
Terrain Statistics:

Original Terrain:
  Height Range: {original_terrain.height_field_raw.min()} ~ {original_terrain.height_field_raw.max()}
  Mean Height: {original_terrain.height_field_raw.mean():.1f}
  Std Height: {original_terrain.height_field_raw.std():.1f}

Stage 1 Terrain:
  Height Range: {stage1_terrain.height_field_raw.min()} ~ {stage1_terrain.height_field_raw.max()}
  Mean Height: {stage1_terrain.height_field_raw.mean():.1f}
  Std Height: {stage1_terrain.height_field_raw.std():.1f}

Valid Standing Areas:
  Total Grid Points: {stage1_terrain.valid_standing_mask.size}
  Valid Points: {np.sum(stage1_terrain.valid_standing_mask)}
  Coverage: {100 * np.sum(stage1_terrain.valid_standing_mask) / stage1_terrain.valid_standing_mask.size:.1f}%

Goals: {len(goals_stage1)} waypoints
    """
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/parkour_terrain_detailed.png', 
                dpi=150, bbox_inches='tight')
    print("详细地形分析图已保存到: parkour_terrain_detailed.png")
    
    # 打印详细分析
    print("\n📊 地形分析结果:")
    print(f"原始地形高度范围: {original_terrain.height_field_raw.min()} ~ {original_terrain.height_field_raw.max()}")
    print(f"阶段1地形高度范围: {stage1_terrain.height_field_raw.min()} ~ {stage1_terrain.height_field_raw.max()}")
    print(f"可站立区域覆盖率: {100 * np.sum(stage1_terrain.valid_standing_mask) / stage1_terrain.valid_standing_mask.size:.1f}%")
    
    # 分析深坑填充效果
    deep_pit_threshold = -50
    original_deep_pits = np.sum(original_terrain.height_field_raw < deep_pit_threshold)
    stage1_deep_pits = np.sum(stage1_terrain.height_field_raw < deep_pit_threshold)
    print(f"原始地形深坑数量: {original_deep_pits}")
    print(f"阶段1地形深坑数量: {stage1_deep_pits}")
    print(f"深坑填充效果: {100 * (1 - stage1_deep_pits / max(original_deep_pits, 1)):.1f}%")
    
    return True

if __name__ == "__main__":
    try:
        success = visualize_parkour_terrain()
        if success:
            print("✅ 地形可视化完成！")
        else:
            print("❌ 地形可视化失败！")
    except Exception as e:
        print(f"❌ 可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()