#!/usr/bin/env python3
"""
创建H1_2机器人的T-pose文件
从URDF文件提取关节信息并生成标准的T-pose姿态
"""

import numpy as np
import torch
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
import os

def create_h1_2_tpose():
    """基于H1的T-pose创建H1_2的T-pose"""
    
    # 先加载H1的T-pose作为模板
    h1_tpose_path = "data/tpose/h1_tpose.npy"
    h1_skeleton_state = SkeletonState.from_file(h1_tpose_path)
    
    print("H1 joint names:", h1_skeleton_state.skeleton_tree.node_names)
    print("H1 parent indices:", h1_skeleton_state.skeleton_tree.parent_indices)
    
    # H1_2的关节名称（基于URDF分析的正确顺序）
    h1_2_joint_names = [
        "pelvis",  # root (index 0)
        "left_hip_yaw_joint",    # index 1, parent: pelvis
        "left_hip_pitch_joint",  # index 2, parent: left_hip_yaw_joint
        "left_hip_roll_joint",   # index 3, parent: left_hip_pitch_joint
        "left_knee_joint",       # index 4, parent: left_hip_roll_joint
        "left_ankle_pitch_joint", # index 5, parent: left_knee_joint
        "left_ankle_roll_joint", # index 6, parent: left_ankle_pitch_joint
        "right_hip_yaw_joint",   # index 7, parent: pelvis
        "right_hip_pitch_joint", # index 8, parent: right_hip_yaw_joint
        "right_hip_roll_joint",  # index 9, parent: right_hip_pitch_joint
        "right_knee_joint",      # index 10, parent: right_hip_roll_joint
        "right_ankle_pitch_joint", # index 11, parent: right_knee_joint
        "right_ankle_roll_joint" # index 12, parent: right_ankle_pitch_joint
    ]
    
    # 基于URDF分析的正确parent indices
    h1_2_parent_indices = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11], dtype=torch.long)
    
    # 基于H1_2 URDF的实际本地变换（从URDF解析得出）
    h1_2_local_translations = torch.tensor([
        [0.0, 0.0, 0.0],         # pelvis (root)
        [0.0, 0.0875, -0.1632],  # left_hip_yaw_joint
        [0.0, 0.0755, 0.0],      # left_hip_pitch_joint
        [0.0, 0.0, 0.0],         # left_hip_roll_joint
        [0.0, 0.0, -0.4],        # left_knee_joint
        [0.0, 0.0, -0.4],        # left_ankle_pitch_joint
        [0.0, 0.0, -0.02],       # left_ankle_roll_joint
        [0.0, -0.0875, -0.1632], # right_hip_yaw_joint
        [0.0, -0.0755, 0.0],     # right_hip_pitch_joint
        [0.0, 0.0, 0.0],         # right_hip_roll_joint
        [0.0, 0.0, -0.4],        # right_knee_joint
        [0.0, 0.0, -0.4],        # right_ankle_pitch_joint
        [0.0, 0.0, -0.02],       # right_ankle_roll_joint
    ], dtype=torch.float32)
    
    # 本地旋转（T-pose时都是单位四元数）
    h1_2_local_rotations = torch.zeros((len(h1_2_joint_names), 4), dtype=torch.float32)
    h1_2_local_rotations[:, 3] = 1.0  # w=1, x=y=z=0 (单位四元数)
    
    # 创建一个新的骨架树字典（按照正确的格式）
    h1_2_skeleton_dict = {
        'node_names': h1_2_joint_names,
        'parent_indices': {
            'arr': h1_2_parent_indices.numpy(),
            'context': {'dtype': 'int64'}
        },
        'local_translation': {
            'arr': h1_2_local_translations.numpy(),
            'context': {'dtype': 'float32'}
        }
    }
    
    # 从字典创建骨架树
    h1_2_skeleton_tree = SkeletonTree.from_dict(h1_2_skeleton_dict)
    
    # 创建T-pose状态（所有关节角度为0）
    h1_2_local_rotation = torch.zeros((len(h1_2_joint_names), 4), dtype=torch.float32)
    h1_2_local_rotation[:, 3] = 1.0  # w=1, x=y=z=0 (单位四元数)
    
    # H1_2的根位置（稍微低一些，适合H1_2的高度）
    h1_2_root_translation = torch.tensor([0.0, 0.0, 0.75], dtype=torch.float32)
    
    # 创建骨架状态
    h1_2_skeleton_state = SkeletonState.from_rotation_and_root_translation(
        h1_2_skeleton_tree, 
        h1_2_local_rotation, 
        h1_2_root_translation, 
        is_local=True
    )
    
    # 保存T-pose文件
    output_path = "data/tpose/h1_2_tpose.npy"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h1_2_skeleton_state.to_file(output_path)
    
    print(f"H1_2 T-pose已保存到: {output_path}")
    print(f"关节数量: {len(h1_2_joint_names)}")
    print(f"关节名称: {h1_2_joint_names}")
    
    return output_path

if __name__ == "__main__":
    create_h1_2_tpose()