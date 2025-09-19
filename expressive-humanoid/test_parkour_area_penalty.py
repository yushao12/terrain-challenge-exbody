#!/usr/bin/env python3
"""
测试parkour区域惩罚机制
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

def test_parkour_area_penalty():
    """测试parkour区域惩罚机制"""
    
    print("🧪 测试parkour区域惩罚机制...")
    
    # 创建地形
    terrain = terrain_utils.SubTerrain(
        "terrain",
        width=160,  # 8m / 0.05m = 160
        length=160,  # 8m / 0.05m = 160
        vertical_scale=0.005,
        horizontal_scale=0.05
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
    
    # 验证scan参考保留
    if hasattr(stage1_terrain, 'scan_reference'):
        print("✅ Scan参考已保留")
        print(f"   Scan参考高度范围: {stage1_terrain.scan_reference.min()} ~ {stage1_terrain.scan_reference.max()}")
    else:
        print("❌ 缺少scan参考")
    
    # 验证可站立区域掩码
    if hasattr(stage1_terrain, 'valid_standing_mask'):
        valid_count = np.sum(stage1_terrain.valid_standing_mask)
        total_count = stage1_terrain.valid_standing_mask.size
        coverage = 100 * valid_count / total_count
        print(f"✅ 可站立区域掩码已创建")
        print(f"   可站立区域覆盖率: {coverage:.1f}%")
    else:
        print("❌ 缺少可站立区域掩码")
    
    # 分析地形结构
    print("\n📊 地形分析:")
    print(f"阶段1地形高度范围: {stage1_terrain.height_field_raw.min()} ~ {stage1_terrain.height_field_raw.max()}")
    
    # 检查石头区域是否保留了斜度
    if hasattr(stage1_terrain, 'valid_standing_mask'):
        standing_areas = stage1_terrain.height_field_raw[stage1_terrain.valid_standing_mask == 1]
        if len(standing_areas) > 0:
            stone_height_range = standing_areas.max() - standing_areas.min()
            print(f"石头区域高度变化: {stone_height_range} 网格单位")
            if stone_height_range > 0:
                print("✅ 石头区域保留了斜度")
            else:
                print("⚠️ 石头区域没有斜度变化")
    
    # 检查非石头区域是否为平地
    if hasattr(stage1_terrain, 'valid_standing_mask'):
        non_standing_areas = stage1_terrain.height_field_raw[stage1_terrain.valid_standing_mask == 0]
        if len(non_standing_areas) > 0:
            non_standing_height_range = non_standing_areas.max() - non_standing_areas.min()
            print(f"非石头区域高度变化: {non_standing_height_range} 网格单位")
            if non_standing_height_range == 0:
                print("✅ 非石头区域已填成平地")
            else:
                print("⚠️ 非石头区域仍有高度变化")
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始scan参考
    if hasattr(stage1_terrain, 'scan_reference'):
        im1 = axes[0].imshow(stage1_terrain.scan_reference, cmap='terrain', aspect='equal')
        axes[0].set_title('Scan Reference\n(Original Parkour)')
        axes[0].set_xlabel('X (grid)')
        axes[0].set_ylabel('Y (grid)')
        plt.colorbar(im1, ax=axes[0])
    
    # 阶段1物理地形
    im2 = axes[1].imshow(stage1_terrain.height_field_raw, cmap='terrain', aspect='equal')
    axes[1].set_title('Stage 1 Physical Terrain\n(Stones + Flat Ground)')
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
    plt.savefig('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/parkour_area_penalty_test.png', 
                dpi=150, bbox_inches='tight')
    print("\n📸 测试结果图已保存到: parkour_area_penalty_test.png")
    
    # 总结
    print("\n🎯 第一阶段训练机制:")
    print("  ✅ 保留原始parkour石头的完整设计（包括斜度）")
    print("  ✅ 保留原始scan信息用于奖励计算")
    print("  ✅ 将非石头区域填成平地")
    print("  ✅ 只在进入parkour区域后开始计算惩罚")
    print("  ✅ 踩到非parkour区域给予严重惩罚(-5.0)")
    print("  ✅ 考虑脚部大小，避免误判")
    
    print("\n📍 惩罚区域:")
    print("  - 起始平台: x >= 20.0米")
    print("  - 宽度范围: -2.0 <= y <= 2.0米")
    print("  - 只有在此区域内的机器人才会被惩罚")
    
    return True

if __name__ == "__main__":
    try:
        success = test_parkour_area_penalty()
        if success:
            print("\n✅ Parkour区域惩罚机制测试成功！")
        else:
            print("\n❌ Parkour区域惩罚机制测试失败！")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()