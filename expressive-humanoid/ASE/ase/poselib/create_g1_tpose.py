#!/usr/bin/env python3
"""
创建G1机器人的T-pose文件
从URDF文件提取关节信息并生成标准的T-pose姿态
"""

import numpy as np
import torch
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
import os

def create_g1_tpose():
    """基于H1的T-pose创建G1的T-pose"""
    
    # 先加载H1的T-pose作为模板
    h1_tpose_path = "data/tpose/h1_tpose.npy"
    h1_skeleton_state = SkeletonState.from_file(h1_tpose_path)
    
    print("H1 joint names:", h1_skeleton_state.skeleton_tree.node_names)
    print("H1 parent indices:", h1_skeleton_state.skeleton_tree.parent_indices)
    
    # G1的关节名称（只包含12个DOF的下半身关节）
    g1_joint_names = [
        "pelvis",  # root
        "left_hip_pitch_joint",
        "left_hip_roll_joint", 
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint", 
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint"
    ]
    
    # 基于H1的骨架结构，创建G1的骨架
    # 我们保留H1的骨架结构，但只保留G1需要的关节
    g1_parent_indices = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11], dtype=torch.long)
    
    # 基于URDF的本地变换（从URDF分析得出）
    g1_local_translations = torch.tensor([
        [0.0, 0.0, 0.0],  # pelvis (root)
        [0.0, 0.064452, -0.1027],  # left_hip_pitch_joint
        [0.0, 0.052, -0.030465],  # left_hip_roll_joint  
        [0.025001, 0.0, -0.12412],  # left_hip_yaw_joint
        [-0.078273, 0.0021489, -0.17734],  # left_knee_joint
        [0.0, -9.4445E-05, -0.30001],  # left_ankle_pitch_joint
        [0.0, 0.0, -0.017558],  # left_ankle_roll_joint
        [0.0, -0.064452, -0.1027],  # right_hip_pitch_joint
        [0.0, 0.052, -0.030465],  # right_hip_roll_joint
        [0.025001, 0.0, -0.12412],  # right_hip_yaw_joint
        [-0.078273, 0.0021489, -0.17734],  # right_knee_joint
        [0.0, -9.4445E-05, -0.30001],  # right_ankle_pitch_joint
        [0.0, 0.0, -0.017558],  # right_ankle_roll_joint
    ], dtype=torch.float32)
    
    # 本地旋转（T-pose时都是单位四元数）
    g1_local_rotations = torch.zeros((len(g1_joint_names), 4), dtype=torch.float32)
    g1_local_rotations[:, 3] = 1.0  # w=1, x=y=z=0 (单位四元数)
    
    # 创建一个新的骨架树字典（按照正确的格式）
    g1_skeleton_dict = {
        'node_names': g1_joint_names,
        'parent_indices': {
            'arr': g1_parent_indices.numpy(),
            'context': {'dtype': 'int64'}
        },
        'local_translation': {
            'arr': g1_local_translations.numpy(),
            'context': {'dtype': 'float32'}
        }
    }
    
    # 从字典创建骨架树
    g1_skeleton_tree = SkeletonTree.from_dict(g1_skeleton_dict)
    
    # 创建T-pose状态（所有关节角度为0）
    g1_local_rotation = torch.zeros((len(g1_joint_names), 4), dtype=torch.float32)
    g1_local_rotation[:, 3] = 1.0  # w=1, x=y=z=0 (单位四元数)
    
    # G1的根位置（稍微低一些，适合G1的高度）
    g1_root_translation = torch.tensor([0.0, 0.0, 0.75], dtype=torch.float32)
    
    # 创建骨架状态
    g1_skeleton_state = SkeletonState.from_rotation_and_root_translation(
        g1_skeleton_tree, 
        g1_local_rotation, 
        g1_root_translation, 
        is_local=True
    )
    
    # 保存T-pose文件
    output_path = "data/tpose/g1_tpose.npy"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    g1_skeleton_state.to_file(output_path)
    
    print(f"G1 T-pose已保存到: {output_path}")
    print(f"关节数量: {len(g1_joint_names)}")
    print(f"关节名称: {g1_joint_names}")
    
    return output_path

if __name__ == "__main__":
    create_g1_tpose()