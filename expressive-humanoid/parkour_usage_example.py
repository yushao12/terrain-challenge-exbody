#!/usr/bin/env python3
"""
Parkour两阶段训练使用示例
"""

import sys
import os

# 添加路径
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym')

def show_usage_example():
    """展示如何使用新的parkour两阶段训练"""
    
    print("=" * 60)
    print("🏃‍♂️ Parkour两阶段训练使用指南")
    print("=" * 60)
    
    print("\n📋 已实现的功能：")
    print("✅ 1. 新的parkour地形变体：parkour_training_stage1")
    print("✅ 2. 保留原始parkour的scan信息")
    print("✅ 3. 物理地形填充为平地（阶段1）")
    print("✅ 4. 脚部接触惩罚系统")
    print("✅ 5. 智能脚部大小考虑")
    
    print("\n🔧 配置说明：")
    print("在 h1_mimic_config.py 中：")
    print("""
    class terrain(LeggedRobotCfg.terrain):
        # 两阶段训练配置
        two_stage_training = True
        training_stage = 1  # 1: 平地训练, 2: 真实地形训练
        stage1_duration = 1000000  # 阶段一训练步数
        stage2_duration = 1000000  # 阶段二训练步数
        foot_size_tolerance = 0.1  # 脚部大小容忍度（米）
    
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            # parkour训练相关奖励
            feet_parkour_penalty = -1.0  # 脚部接触非parkour区域的惩罚权重
    """)
    
    print("\n🎯 训练阶段切换：")
    print("阶段1（平地训练）：")
    print("  - 物理地形：平地，便于学习基本步态")
    print("  - Scan信息：原始parkour地形")
    print("  - 奖励：脚部接触惩罚，引导学习正确落脚点")
    print("  - 目标：学会基本的parkour步态模式")
    
    print("\n阶段2（真实地形训练）：")
    print("  - 物理地形：真实parkour地形")
    print("  - Scan信息：原始parkour地形")
    print("  - 奖励：正常奖励系统")
    print("  - 目标：在真实地形上微调和最终训练")
    
    print("\n🚀 使用方法：")
    print("1. 修改 combine_config.py 中的 proportions：")
    print("   proportions = [")
    print("       (\"single\", 1, 1),  # 使用parkour_training_stage1 (索引1)")
    print("   ]")
    
    print("\n2. 设置训练阶段：")
    print("   # 阶段1训练")
    print("   cfg.terrain.training_stage = 1")
    print("   ")
    print("   # 阶段2训练")
    print("   cfg.terrain.training_stage = 2")
    
    print("\n3. 运行训练：")
    print("   python train.py --task h1_mimic --num_envs 4096")
    
    print("\n📊 监控指标：")
    print("- feet_parkour_penalty: 脚部接触惩罚（阶段1）")
    print("- 地形对比图：parkour_terrain_comparison.png")
    print("- 可站立区域数量：4855个网格点")
    
    print("\n🔍 技术细节：")
    print("- 地形掩码：valid_standing_mask 标记可站立区域")
    print("- Scan参考：scan_reference 保存原始parkour地形")
    print("- 脚部大小：考虑0.1米半径的容忍度")
    print("- 接触检测：基于contact_filt和地形掩码")
    
    print("\n⚠️  注意事项：")
    print("1. 确保在阶段1时启用two_stage_training=True")
    print("2. 脚部大小容忍度可根据机器人实际尺寸调整")
    print("3. 惩罚权重可根据训练效果调整")
    print("4. 阶段切换时机可根据训练进度调整")
    
    print("\n" + "=" * 60)
    print("🎉 实现完成！可以开始两阶段parkour训练了！")
    print("=" * 60)

if __name__ == "__main__":
    show_usage_example()