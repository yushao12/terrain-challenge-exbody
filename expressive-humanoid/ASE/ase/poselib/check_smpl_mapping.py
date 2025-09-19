#!/usr/bin/env python3
"""
检查SMPL关节映射和AMASS数据格式
"""

import numpy as np
import torch
from amass_importer import create_smpl_skeleton_tree

def check_smpl_joint_order():
    """检查SMPL关节顺序"""
    print("=== SMPL关节顺序检查 ===")
    
    # 我们的SMPL关节名称
    our_joint_names = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 
        'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 
        'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 
        'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
    ]
    
    # 标准SMPL关节名称 (从SMPL论文)
    standard_smpl_names = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 
        'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 
        'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 
        'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
    ]
    
    print(f"我们的关节数量: {len(our_joint_names)}")
    print(f"标准SMPL关节数量: {len(standard_smpl_names)}")
    
    # 检查是否匹配
    if our_joint_names == standard_smpl_names:
        print("✓ 关节名称匹配")
    else:
        print("✗ 关节名称不匹配")
        for i, (our, std) in enumerate(zip(our_joint_names, standard_smpl_names)):
            if our != std:
                print(f"  关节 {i}: 我们='{our}', 标准='{std}'")

def check_amass_data_format():
    """检查AMASS数据格式"""
    print("\n=== AMASS数据格式检查 ===")
    
    try:
        # 加载AMASS数据
        data_path = "data/amass_test/B1_-_stand_to_walk_stageii.npz"
        data = np.load(data_path)
        
        print(f"AMASS文件: {data_path}")
        print(f"可用字段: {list(data.keys())}")
        
        # 检查poses字段
        if 'poses' in data:
            poses = data['poses']
            print(f"poses形状: {poses.shape}")
            print(f"poses数据类型: {poses.dtype}")
            
            # 检查是否是72维 (24关节 × 3)
            if poses.shape[1] == 72:
                print("✓ poses维度正确 (24关节 × 3)")
            else:
                print(f"✗ poses维度错误: 期望72, 实际{poses.shape[1]}")
                
            # 检查是否是轴角格式
            # 轴角格式应该是 [angle * axis_x, angle * axis_y, angle * axis_z]
            sample_pose = poses[0]  # 第一帧
            print(f"第一帧pose范围: [{sample_pose.min():.3f}, {sample_pose.max():.3f}]")
            
        # 检查trans字段
        if 'trans' in data:
            trans = data['trans']
            print(f"trans形状: {trans.shape}")
            print(f"trans数据类型: {trans.dtype}")
            
        # 检查betas字段
        if 'betas' in data:
            betas = data['betas']
            print(f"betas形状: {betas.shape}")
            
    except Exception as e:
        print(f"加载AMASS数据失败: {e}")

def check_axis_angle_conversion():
    """检查轴角到四元数的转换"""
    print("\n=== 轴角转换检查 ===")
    
    # 测试简单的轴角转换
    test_axis_angle = np.array([0, 0, np.pi/2])  # 绕Z轴旋转90度
    
    print(f"测试轴角: {test_axis_angle}")
    
    # 计算角度和轴
    angle = np.linalg.norm(test_axis_angle)
    axis = test_axis_angle / angle if angle > 0 else np.array([0, 0, 1])
    
    print(f"计算得到 - 角度: {angle:.3f}, 轴: {axis}")
    
    # 转换为四元数
    from poselib.core.rotation3d import quat_from_angle_axis
    quat = quat_from_angle_axis(
        torch.tensor([angle]), 
        torch.tensor([axis])
    )
    
    print(f"转换后的四元数: {quat.numpy()}")
    
    # 验证四元数是否正确 (绕Z轴旋转90度)
    expected_quat = np.array([0, 0, 0.7071, 0.7071])  # [x, y, z, w]
    print(f"期望的四元数: {expected_quat}")
    
    diff = np.abs(quat.numpy() - expected_quat).max()
    if diff < 0.01:
        print("✓ 轴角转换正确")
    else:
        print(f"✗ 轴角转换错误, 最大差异: {diff}")

def check_skeleton_tree():
    """检查骨架树结构"""
    print("\n=== 骨架树结构检查 ===")
    
    skeleton_tree = create_smpl_skeleton_tree()
    
    print(f"关节数量: {len(skeleton_tree)}")
    print(f"关节名称: {skeleton_tree.node_names}")
    print(f"父关节索引: {skeleton_tree.parent_indices}")
    
    # 检查根关节
    root_idx = (skeleton_tree.parent_indices == -1).nonzero().item()
    print(f"根关节: {skeleton_tree.node_names[root_idx]} (索引: {root_idx})")
    
    # 检查关键关节的父子关系
    key_joints = ['Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee']
    for joint in key_joints:
        if joint in skeleton_tree.node_names:
            idx = skeleton_tree.node_names.index(joint)
            parent_idx = skeleton_tree.parent_indices[idx]
            parent_name = skeleton_tree.node_names[parent_idx] if parent_idx >= 0 else 'None'
            print(f"{joint} -> {parent_name}")

def main():
    """主函数"""
    print("=== AMASS数据格式和映射检查 ===")
    
    check_smpl_joint_order()
    check_amass_data_format()
    check_axis_angle_conversion()
    check_skeleton_tree()
    
    print("\n=== 检查完成 ===")

if __name__ == "__main__":
    main() 