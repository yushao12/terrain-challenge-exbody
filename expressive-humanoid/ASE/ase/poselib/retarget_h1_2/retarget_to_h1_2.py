#!/usr/bin/env python3

import os
import sys
import numpy as np
import yaml
from pathlib import Path

# 添加ASE路径
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/ASE/ase')
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/ASE/ase/utils')

def retarget_motions_to_h1_2():
    """将motion数据重定向到H1_2机器人"""
    
    # 输入和输出路径
    input_motion_dir = Path("/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/ASE/ase/poselib/data/npy/npy")
    output_motion_dir = Path("/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/ASE/ase/poselib/data/npy_h1_2")
    
    # 创建输出目录
    output_motion_dir.mkdir(parents=True, exist_ok=True)
    
    # H1_2的关节映射（基于URDF结构）
    # H1_2有12个DOF，与G1结构相似
    h1_2_joint_mapping = {
        # 左腿
        'left_hip_yaw': 0,
        'left_hip_pitch': 1, 
        'left_hip_roll': 2,
        'left_knee': 3,
        'left_ankle_pitch': 4,
        'left_ankle_roll': 5,
        # 右腿
        'right_hip_yaw': 6,
        'right_hip_pitch': 7,
        'right_hip_roll': 8,
        'right_knee': 9,
        'right_ankle_pitch': 10,
        'right_ankle_roll': 11,
    }
    
    print("H1_2 Joint Mapping:")
    for joint_name, idx in h1_2_joint_mapping.items():
        print(f"  {joint_name}: {idx}")
    
    # 处理motion文件
    if input_motion_dir.exists():
        motion_files = list(input_motion_dir.glob("*.npy"))
        print(f"Found {len(motion_files)} motion files to retarget")
        
        for motion_file in motion_files:
            print(f"Processing {motion_file.name}...")
            
            try:
                # 加载原始motion数据
                motion_data = np.load(motion_file, allow_pickle=True)
                print(f"  Original shape: {motion_data.shape}")
                
                # 检查数据格式
                if motion_data.dtype == object and hasattr(motion_data, 'item'):
                    motion_dict = motion_data.item()
                    print(f"  Motion keys: {list(motion_dict.keys())}")
                    
                    # 检查rotation数据的结构
                    if 'rotation' in motion_dict:
                        rotation_data = motion_dict['rotation']
                        print(f"  Rotation type: {type(rotation_data)}")
                        if isinstance(rotation_data, dict):
                            print(f"  Rotation keys: {list(rotation_data.keys())}")
                            # 获取第一个关节的数据来了解格式
                            first_joint = list(rotation_data.keys())[0]
                            first_data = rotation_data[first_joint]
                            print(f"  First joint ({first_joint}) data shape: {first_data.shape}")
                    
                    # 暂时直接复制数据，实际重定向逻辑需要根据具体的关节映射来实现
                    retargeted_data = motion_data
                    
                    # 保存重定向后的数据
                    output_file = output_motion_dir / motion_file.name
                    np.save(output_file, retargeted_data)
                    print(f"  Saved to: {output_file}")
                else:
                    print(f"  Unexpected data format: {motion_data.dtype}")
                
            except Exception as e:
                print(f"  Error processing {motion_file.name}: {e}")
                import traceback
                traceback.print_exc()
    
    else:
        print(f"Input motion directory {input_motion_dir} does not exist!")
    
    print("H1_2 motion retargeting completed!")

if __name__ == "__main__":
    retarget_motions_to_h1_2()