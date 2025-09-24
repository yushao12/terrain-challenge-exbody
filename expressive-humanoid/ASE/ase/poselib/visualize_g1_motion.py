#!/usr/bin/env python3
"""
可视化G1转换后的动作数据
"""

import os
import sys
import numpy as np
from poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.visualization.common import plot_skeleton_motion_interactive

def visualize_g1_motion(motion_file=None):
    """可视化G1转换后的动作"""
    print("=== G1动作可视化 ===")
    
    # 如果没有指定文件，使用默认的测试文件
    if motion_file is None:
        # 选择一个较短的motion文件进行测试
        motion_file = "data/npy/npy/102_09.npy"  # 这个文件比较小，适合测试
    
    if not os.path.exists(motion_file):
        print(f"文件不存在: {motion_file}")
        print("可用的motion文件:")
        
        # 列出一些可用的motion文件
        npy_dir = "data/npy/npy"
        if os.path.exists(npy_dir):
            files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
            files.sort()
            for i, f in enumerate(files[:10]):  # 只显示前10个
                file_path = os.path.join(npy_dir, f)
                file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                print(f"  {i+1}. {f} ({file_size:.1f}MB)")
            if len(files) > 10:
                print(f"  ... 还有 {len(files)-10} 个文件")
        
        return
    
    try:
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
        
        plot_skeleton_motion_interactive(motion, task_name="G1动作可视化")
        
    except Exception as e:
        print(f"加载或可视化动作时出错: {e}")
        import traceback
        traceback.print_exc()

def list_g1_motions():
    """列出所有可用的G1 motion文件"""
    print("=== G1 Motion文件列表 ===")
    
    npy_dir = "data/npy/npy"
    if not os.path.exists(npy_dir):
        print(f"目录不存在: {npy_dir}")
        return
    
    files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    files.sort()
    
    print(f"找到 {len(files)} 个motion文件:")
    for i, f in enumerate(files):
        file_path = os.path.join(npy_dir, f)
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"  {i+1:2d}. {f} ({file_size:6.1f}MB)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_g1_motions()
        else:
            # 使用命令行参数指定的文件
            motion_file = sys.argv[1]
            visualize_g1_motion(motion_file)
    else:
        # 默认行为
        visualize_g1_motion()