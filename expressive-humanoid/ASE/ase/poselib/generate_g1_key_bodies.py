#!/usr/bin/env python3
"""
为G1的motion数据生成key_bodies文件
从现有的motion数据中提取关键关节位置并保存为key_bodies.npy文件
"""

import numpy as np
import os
import sys
from pathlib import Path

# 添加poselib路径
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/ASE/ase/poselib')
from poselib.skeleton.skeleton3d import SkeletonMotion

# G1关键关节索引（基于URDF分析）
G1_KEY_BODY_INDICES = [0, 2, 5, 6, 8, 11, 12]
G1_KEY_BODY_NAMES = [
    'pelvis', 
    'left_hip_pitch_link', 
    'left_knee_link', 
    'left_ankle_pitch_link', 
    'right_hip_pitch_link', 
    'right_knee_link', 
    'right_ankle_pitch_link'
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
    num_key_bodies = len(G1_KEY_BODY_INDICES)
    
    # 初始化关键关节位置数组
    key_body_pos = np.zeros((num_frames, num_key_bodies, 3), dtype=np.float32)
    
    # 获取全局位置
    global_translations = motion_data.global_translation  # (num_frames, num_joints, 3)
    root_translation = motion_data.root_translation  # (num_frames, 3)
    
    for frame_idx in range(num_frames):
        for body_idx, joint_idx in enumerate(G1_KEY_BODY_INDICES):
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

def generate_all_g1_key_bodies():
    """
    为所有G1 motion文件生成key_bodies数据
    """
    # G1 motion数据目录
    g1_motion_dir = Path("/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/ASE/ase/poselib/data/retarget_npy_g1")
    
    if not g1_motion_dir.exists():
        print(f"Error: G1 motion directory not found: {g1_motion_dir}")
        return
    
    # 查找所有motion文件
    motion_files = list(g1_motion_dir.glob("*.npy"))
    motion_files = [f for f in motion_files if not "_key_bodies" in str(f.name)]
    
    print(f"Found {len(motion_files)} motion files to process")
    
    # 处理每个motion文件
    success_count = 0
    for i, motion_file in enumerate(motion_files):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(motion_files)}")
        if process_motion_file(motion_file):
            success_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Total files: {len(motion_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(motion_files) - success_count}")

def verify_generated_key_bodies():
    """
    验证生成的key_bodies文件
    """
    g1_motion_dir = Path("/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/ASE/ase/poselib/data/retarget_npy_g1")
    key_body_files = list(g1_motion_dir.glob("*_key_bodies.npy"))
    
    print(f"\n=== Verification ===")
    print(f"Generated {len(key_body_files)} key_bodies files")
    
    # 检查几个文件的格式
    for i, key_body_file in enumerate(key_body_files[:3]):  # 只检查前3个
        try:
            data = np.load(key_body_file)
            print(f"{key_body_file.name}: shape={data.shape}, dtype={data.dtype}")
            print(f"  Range: min={data.min():.4f}, max={data.max():.4f}")
        except Exception as e:
            print(f"Error loading {key_body_file}: {e}")

if __name__ == "__main__":
    print("=== G1 Key Bodies Generation ===")
    print(f"Key body indices: {G1_KEY_BODY_INDICES}")
    print(f"Key body names: {G1_KEY_BODY_NAMES}")
    
    generate_all_g1_key_bodies()
    verify_generated_key_bodies()
    
    print("\nDone!")