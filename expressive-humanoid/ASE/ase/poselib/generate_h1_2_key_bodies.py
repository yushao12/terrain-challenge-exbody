#!/usr/bin/env python3
"""
为H1_2的motion数据生成key_bodies文件
从现有的motion数据中提取关键关节位置并保存为key_bodies.npy文件
"""

import numpy as np
import os
import sys
from pathlib import Path

# 添加poselib路径
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/ASE/ase/poselib')
from poselib.skeleton.skeleton3d import SkeletonMotion

# H1_2关键关节索引（基于URDF分析）
# H1_2关节顺序：pelvis, left_hip_yaw_joint, left_hip_pitch_joint, left_hip_roll_joint, left_knee_joint, left_ankle_pitch_joint, left_ankle_roll_joint, right_hip_yaw_joint, right_hip_pitch_joint, right_hip_roll_joint, right_knee_joint, right_ankle_pitch_joint, right_ankle_roll_joint
H1_2_KEY_BODY_INDICES = [2, 4, 8, 10]  # left_hip_pitch, left_knee, right_hip_pitch, right_knee
H1_2_KEY_BODY_NAMES = [
    'left_hip_pitch_joint', 
    'left_knee_joint', 
    'left_ankle_roll_joint', 
    'right_hip_pitch_joint', 
    'right_knee_joint', 
    'right_ankle_roll_joint'
]

def extract_key_body_positions(motion_data):
    """
    从motion数据中提取关键关节的局部位置
    
    Args:
        motion_data: SkeletonMotion对象
    
    Returns:
        key_body_pos: (num_frames, num_key_bodies, 3) 关键关节的局部位置
    """
    num_frames = motion_data.tensor.shape[0]
    num_key_bodies = len(H1_2_KEY_BODY_INDICES)
    
    # 初始化关键关节位置数组
    key_body_pos = np.zeros((num_frames, num_key_bodies, 3), dtype=np.float32)
    
    # 获取全局位置
    global_translations = motion_data.global_translation  # (num_frames, num_joints, 3)
    root_translation = motion_data.root_translation  # (num_frames, 3)
    
    for frame_idx in range(num_frames):
        for body_idx, joint_idx in enumerate(H1_2_KEY_BODY_INDICES):
            if joint_idx < global_translations.shape[1]:
                # 获取全局位置
                global_pos = global_translations[frame_idx, joint_idx, :]
                root_pos = root_translation[frame_idx, :]
                
                # 转换为相对于root的局部位置
                local_pos = global_pos - root_pos
                key_body_pos[frame_idx, body_idx, :] = local_pos
            else:
                print(f"Warning: Joint index {joint_idx} out of range for frame {frame_idx}")
    
    return key_body_pos

def process_motion_file(motion_file_path):
    """
    处理单个motion文件，生成对应的key_bodies文件
    
    Args:
        motion_file_path: motion文件路径
    """
    print(f"Processing: {motion_file_path}")
    
    try:
        # 加载motion数据
        motion_data = SkeletonMotion.from_file(str(motion_file_path))
        
        # 提取关键关节位置
        key_body_pos = extract_key_body_positions(motion_data)
        
        # 生成输出文件路径
        motion_name = motion_file_path.stem
        output_dir = motion_file_path.parent
        output_file = output_dir / f"{motion_name}_key_bodies.npy"
        
        # 保存key_bodies数据
        np.save(output_file, key_body_pos)
        
        print(f"  -> Generated: {output_file}")
        print(f"  -> Shape: {key_body_pos.shape}")
        print(f"  -> Frames: {key_body_pos.shape[0]}, Key bodies: {key_body_pos.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"  -> Error processing {motion_file_path}: {e}")
        return False

def generate_all_h1_2_key_bodies(motion_dir):
    """
    为所有H1_2 motion文件生成key_bodies数据
    
    Args:
        motion_dir: motion文件目录路径
    """
    motion_path = Path(motion_dir)
    
    if not motion_path.exists():
        print(f"Error: Motion directory {motion_dir} does not exist")
        return
    
    # 查找所有.npy文件
    motion_files = list(motion_path.glob("*.npy"))
    motion_files = [f for f in motion_files if not str(f.name).endswith("_key_bodies.npy")]
    
    print(f"Found {len(motion_files)} motion files to process")
    print(f"Key body indices: {H1_2_KEY_BODY_INDICES}")
    print(f"Key body names: {H1_2_KEY_BODY_NAMES}")
    
    success_count = 0
    error_count = 0
    
    for motion_file in motion_files:
        if process_motion_file(motion_file):
            success_count += 1
        else:
            error_count += 1
    
    print(f"\nKey body generation completed!")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed to process: {error_count} files")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate key body data for H1_2 motions')
    parser.add_argument('--motion_dir', type=str, 
                       default='/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/ASE/ase/poselib/data/retarget_npy_h1_2',
                       help='Directory containing motion .npy files')
    
    args = parser.parse_args()
    generate_all_h1_2_key_bodies(args.motion_dir)

if __name__ == "__main__":
    main()