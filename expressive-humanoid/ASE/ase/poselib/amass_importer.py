#!/usr/bin/env python3
"""
AMASS数据导入脚本
将AMASS SMPL数据转换为SkeletonMotion格式
"""

import os
import numpy as np
import torch
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from poselib.core.rotation3d import quat_from_angle_axis, quat_identity_like, quat_mul

def create_smpl_skeleton_tree():
    """创建SMPL骨架树"""
    # SMPL的24个关节名称
    smpl_joint_names = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 
        'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 
        'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 
        'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
    ]
    
    # SMPL的父关节索引 (从SMPL模型定义)
    smpl_parent_indices = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 
        13, 14, 16, 17, 18, 19, 20, 21
    ]
    
    # 简化的局部位置 (实际应该从SMPL模型获取)
    local_translation = np.zeros((24, 3), dtype=np.float32)
    # 这里可以设置更准确的局部位置，暂时用零向量
    
    return SkeletonTree(
        smpl_joint_names,
        torch.from_numpy(np.array(smpl_parent_indices, dtype=np.int32)),
        torch.from_numpy(local_translation)
    )

def convert_amass_to_skeleton_motion(amass_data, fps=60):
    """
    将AMASS SMPL-X数据转换为SkeletonMotion (SMPL格式)
    
    Args:
        amass_data: 包含'pose_body', 'root_orient', 'trans'的字典
        fps: 帧率
    
    Returns:
        SkeletonMotion对象
    """
    pose_body = amass_data['pose_body']  # [T, 63] - SMPL-X body pose参数
    root_orient = amass_data['root_orient']  # [T, 3] - 根关节方向
    trans = amass_data['trans']  # [T, 3] - 根关节位置
    
    # 创建SMPL骨架树
    skeleton_tree = create_smpl_skeleton_tree()
    
    # 将SMPL-X的21个body关节转换为SMPL的24个关节
    # SMPL-X: 21个body关节 + 根关节 = 22个关节
    # SMPL: 24个关节 (包括根关节)
    
    # 组合pose_body和root_orient
    # pose_body: [T, 63] -> [T, 21, 3]
    # root_orient: [T, 3] -> [T, 1, 3]
    pose_body_reshaped = pose_body.reshape(-1, 21, 3)  # [T, 21, 3]
    root_orient_reshaped = root_orient.reshape(-1, 1, 3)  # [T, 1, 3]
    
    # 合并为22个关节的pose
    poses_22 = np.concatenate([root_orient_reshaped, pose_body_reshaped], axis=1)  # [T, 22, 3]
    
    # 现在需要将22个关节映射到24个关节
    # 这里我们简单地将缺失的关节设为0旋转
    T = poses_22.shape[0]
    poses_24 = np.zeros((T, 24, 3), dtype=poses_22.dtype)
    
    # 映射关节 (这里需要根据SMPL和SMPL-X的关节对应关系)
    # 暂时使用简单映射：前22个关节直接对应，后2个设为0
    poses_24[:, :22, :] = poses_22
    
    # 添加绕x轴旋转九十度的变换
    # 对每个时间步的根关节应用这个旋转
    for t in range(poses_24.shape[0]):
        # 将根关节的轴角转换为四元数
        root_angle = np.linalg.norm(poses_24[t, 0, :])
        if root_angle > 1e-8:
            root_axis = poses_24[t, 0, :] / root_angle
            root_quat = quat_from_angle_axis(
                torch.tensor([root_angle], dtype=torch.float32),
                torch.tensor([root_axis], dtype=torch.float32)
            )
        else:
            root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        
        # 创建x轴旋转90度的四元数 [cos(π/4), sin(π/4), 0, 0]
        x_rot_quat = torch.tensor([[0.7071068, 0.7071068, 0.0, 0.0]], dtype=torch.float32)
        
        # 组合旋转：先应用x轴旋转，再应用原始根关节旋转
        combined_quat = quat_mul(x_rot_quat, root_quat)
        
        # 将四元数转换回轴角表示
        quat_array = combined_quat.numpy()[0]
        
        # 四元数转轴角
        w, x, y, z = quat_array
        if abs(w) >= 1.0:
            angle_val = 0.0
            axis_val = np.array([1.0, 0.0, 0.0])
        else:
            angle_val = 2.0 * np.arccos(abs(w))
            if angle_val > 1e-8:
                axis_val = np.array([x, y, z]) / np.sin(angle_val / 2.0)
                if w < 0:
                    axis_val = -axis_val
            else:
                axis_val = np.array([1.0, 0.0, 0.0])
        
        poses_24[t, 0, :] = angle_val * axis_val
    
    # 计算轴角幅度和方向
    angle = np.linalg.norm(poses_24, axis=-1, keepdims=True)
    axis = poses_24 / (angle + 1e-8)
    
    # 转换为四元数
    poses_quat = quat_from_angle_axis(
        torch.from_numpy(angle.squeeze(-1)).float(),
        torch.from_numpy(axis).float()
    )
    
    # 创建SkeletonState
    skeleton_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=skeleton_tree,
        r=poses_quat,  # [T, 24, 4] 四元数
        t=torch.from_numpy(trans).float(),  # [T, 3] 根位置
        is_local=True
    )
    
    # 创建SkeletonMotion
    motion = SkeletonMotion.from_skeleton_state(
        skeleton_state=skeleton_state,
        fps=fps
    )
    
    return motion

def process_amass_file(input_path, output_path):
    """
    处理单个AMASS文件
    
    Args:
        input_path: AMASS文件路径 (.npz)
        output_path: 输出文件路径 (.npy)
    """
    print(f"Processing {input_path}...")
    
    # 加载AMASS数据
    if input_path.endswith('.npz'):
        data = np.load(input_path, allow_pickle=True)
        amass_data = {
            'pose_body': data['pose_body'],
            'root_orient': data['root_orient'],
            'trans': data['trans']
        }
    else:
        raise ValueError("Input file must be .npz format")
    
    # 转换为SkeletonMotion
    motion = convert_amass_to_skeleton_motion(amass_data)
    
    # 保存为.npy格式
    motion.to_file(output_path)
    print(f"Saved to {output_path}")
    
    return motion

def main():
    """主函数"""
    # 测试目录
    test_dir = "data/amass_test"
    
    # 检查是否有AMASS文件
    amass_files = [f for f in os.listdir(test_dir) if f.endswith('.npz')]
    
    if not amass_files:
        print(f"No AMASS files found in {test_dir}")
        print("Please place your AMASS .npz files in this directory")
        return
    
    # 处理所有AMASS文件
    for amass_file in amass_files:
        input_path = os.path.join(test_dir, amass_file)
        output_path = os.path.join(test_dir, amass_file.replace('.npz', '.npy'))
        
        try:
            motion = process_amass_file(input_path, output_path)
            print(f"Successfully processed {amass_file}")
            print(f"Motion shape: {motion.tensor.shape}")
            print(f"FPS: {motion.fps}")
        except Exception as e:
            print(f"Error processing {amass_file}: {e}")

if __name__ == "__main__":
    main() 