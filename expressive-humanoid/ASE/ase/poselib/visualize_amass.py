#!/usr/bin/env python3
"""
可视化AMASS转换后的动作数据
"""

import os
from poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.visualization.common import plot_skeleton_motion_interactive

def visualize_amass_motion():
    """可视化AMASS转换后的动作"""
    print("=== AMASS动作可视化 ===")
    
    # 转换后的文件路径
    motion_file = "data/amass_test/B1_-_stand_to_walk_stageii.npy"
    
    if not os.path.exists(motion_file):
        print(f"文件不存在: {motion_file}")
        print("请先运行 amass_importer.py 转换AMASS数据")
        return
    
    # 加载动作数据
    print(f"加载动作文件: {motion_file}")
    motion = SkeletonMotion.from_file(motion_file)
    
    print(f"动作信息:")
    print(f"  帧数: {motion.tensor.shape[0]}")
    print(f"  数据维度: {motion.tensor.shape[1]}")
    print(f"  FPS: {motion.fps}")
    print(f"  时长: {motion.tensor.shape[0] / motion.fps:.2f}秒")
    print(f"  关节数: {len(motion.skeleton_tree.node_names)}")
    print(f"  关节名称: {motion.skeleton_tree.node_names}")
    
    # 可视化动作
    print("\n开始可视化...")
    print("使用鼠标和键盘控制:")
    print("  - 鼠标: 旋转视角")
    print("  - 滚轮: 缩放")
    print("  - 空格: 播放/暂停")
    print("  - 左右箭头: 逐帧播放")
    print("  - ESC: 退出")
    
    plot_skeleton_motion_interactive(motion, task_name="AMASS动作可视化")

if __name__ == "__main__":
    visualize_amass_motion() 