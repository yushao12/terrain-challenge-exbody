#!/usr/bin/env python3
"""
使用legged_gym的play功能来可视化G1机器人
"""

import os
import sys
import argparse

def play_g1_with_motion(motion_id="015-05", exptid="015-05"):
    """使用指定的motion数据来play G1"""
    
    # 切换到legged_gym目录
    os.chdir("/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym")
    
    # 构建命令
    cmd = f"""
    python legged_gym/scripts/play.py \\
        --task g1_mimic_amp \\
        --motion-id {motion_id} \\
        --exptid {exptid} \\
        --proj_name g1_mimic_amp \\
        --device cuda:0 \\
        --headless 0 \\
        --num_threads 1
    """
    
    print("=== 启动G1可视化 ===")
    print(f"Motion ID: {motion_id}")
    print(f"Experiment ID: {exptid}")
    print(f"执行命令: {cmd.strip()}")
    
    # 执行命令
    os.system(cmd.strip())

def list_available_motions():
    """列出可用的motion文件"""
    print("=== 可用的G1 Motion文件 ===")
    
    npy_dir = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/ASE/ase/poselib/data/npy/npy"
    if not os.path.exists(npy_dir):
        print(f"目录不存在: {npy_dir}")
        return
    
    files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    files.sort()
    
    print(f"找到 {len(files)} 个motion文件:")
    for i, f in enumerate(files[:20]):  # 只显示前20个
        file_path = os.path.join(npy_dir, f)
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        motion_id = f.replace('.npy', '')
        print(f"  {i+1:2d}. {motion_id} ({file_size:6.1f}MB)")
    
    if len(files) > 20:
        print(f"  ... 还有 {len(files)-20} 个文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play G1 with motion data")
    parser.add_argument("--motion-id", default="015-05", help="Motion ID (e.g., 015-05)")
    parser.add_argument("--exptid", default="015-05", help="Experiment ID")
    parser.add_argument("--list", action="store_true", help="List available motions")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_motions()
    else:
        play_g1_with_motion(args.motion_id, args.exptid)