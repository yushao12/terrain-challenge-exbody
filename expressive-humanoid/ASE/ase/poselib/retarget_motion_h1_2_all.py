from isaacgym.torch_utils import *
import torch
import json
import numpy as np
import sys
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
import yaml
import os
import multiprocessing
from tqdm import tqdm

"""
H1_2版本的motion retarget脚本
从CMU motion数据retarget到H1_2机器人
只处理下半身12个DOF的关节
"""

VISUALIZE = False

def project_joints_h1_2(motion):
    """H1_2版本的关节投影，只处理下半身关节"""
    # 获取H1_2的关节ID（基于H1_2 tpose的正确关节名称）
    # H1_2的关节顺序：pelvis, left_hip_yaw_joint, left_hip_pitch_joint, left_hip_roll_joint, left_knee_joint, left_ankle_pitch_joint, left_ankle_roll_joint, right_hip_yaw_joint, right_hip_pitch_joint, right_hip_roll_joint, right_knee_joint, right_ankle_pitch_joint, right_ankle_roll_joint
    right_thigh_id = motion.skeleton_tree._node_indices["right_hip_pitch_joint"]  # index 8
    right_shin_id = motion.skeleton_tree._node_indices["right_knee_joint"]        # index 10
    right_foot_id = motion.skeleton_tree._node_indices["right_ankle_roll_joint"]  # index 12
    left_thigh_id = motion.skeleton_tree._node_indices["left_hip_pitch_joint"]    # index 2
    left_shin_id = motion.skeleton_tree._node_indices["left_knee_joint"]          # index 4
    left_foot_id = motion.skeleton_tree._node_indices["left_ankle_roll_joint"]    # index 6
    
    device = motion.global_translation.device

    # 处理右腿
    right_thigh_pos = motion.global_translation[..., right_thigh_id, :]
    right_shin_pos = motion.global_translation[..., right_shin_id, :]
    right_foot_pos = motion.global_translation[..., right_foot_id, :]
    right_hip_rot = motion.local_rotation[..., right_thigh_id, :]
    right_knee_rot = motion.local_rotation[..., right_shin_id, :]
    
    right_leg_delta0 = right_thigh_pos - right_shin_pos
    right_leg_delta1 = right_foot_pos - right_shin_pos
    right_leg_delta0 = right_leg_delta0 / torch.norm(right_leg_delta0, dim=-1, keepdim=True)
    right_leg_delta1 = right_leg_delta1 / torch.norm(right_leg_delta1, dim=-1, keepdim=True)
    right_knee_dot = torch.sum(-right_leg_delta0 * right_leg_delta1, dim=-1)
    right_knee_dot = torch.clamp(right_knee_dot, -1.0, 1.0)
    right_knee_theta = torch.acos(right_knee_dot)
    right_knee_q = quat_from_angle_axis(torch.abs(right_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    right_knee_local_dir = motion.skeleton_tree.local_translation[right_foot_id]
    right_knee_local_dir = right_knee_local_dir / torch.norm(right_knee_local_dir)
    right_knee_local_dir_tile = torch.tile(right_knee_local_dir.unsqueeze(0), [right_knee_rot.shape[0], 1])
    right_knee_local_dir0 = quat_rotate(right_knee_rot, right_knee_local_dir_tile)
    right_knee_local_dir1 = quat_rotate(right_knee_q, right_knee_local_dir_tile)
    right_leg_dot = torch.sum(right_knee_local_dir0 * right_knee_local_dir1, dim=-1)
    right_leg_dot = torch.clamp(right_leg_dot, -1.0, 1.0)
    right_leg_theta = torch.acos(right_leg_dot)
    right_leg_theta = torch.where(right_knee_local_dir0[..., 1] >= 0, right_leg_theta, -right_leg_theta)
    right_leg_q = quat_from_angle_axis(right_leg_theta, right_knee_local_dir.unsqueeze(0))
    right_hip_rot = quat_mul(right_hip_rot, right_leg_q)
    
    # 处理左腿（对称处理）
    left_thigh_pos = motion.global_translation[..., left_thigh_id, :]
    left_shin_pos = motion.global_translation[..., left_shin_id, :]
    left_foot_pos = motion.global_translation[..., left_foot_id, :]
    left_hip_rot = motion.local_rotation[..., left_thigh_id, :]
    left_knee_rot = motion.local_rotation[..., left_shin_id, :]
    
    left_leg_delta0 = left_thigh_pos - left_shin_pos
    left_leg_delta1 = left_foot_pos - left_shin_pos
    left_leg_delta0 = left_leg_delta0 / torch.norm(left_leg_delta0, dim=-1, keepdim=True)
    left_leg_delta1 = left_leg_delta1 / torch.norm(left_leg_delta1, dim=-1, keepdim=True)
    left_knee_dot = torch.sum(-left_leg_delta0 * left_leg_delta1, dim=-1)
    left_knee_dot = torch.clamp(left_knee_dot, -1.0, 1.0)
    left_knee_theta = torch.acos(left_knee_dot)
    left_knee_q = quat_from_angle_axis(torch.abs(left_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    left_knee_local_dir = motion.skeleton_tree.local_translation[left_foot_id]
    left_knee_local_dir = left_knee_local_dir / torch.norm(left_knee_local_dir)
    left_knee_local_dir_tile = torch.tile(left_knee_local_dir.unsqueeze(0), [left_knee_rot.shape[0], 1])
    left_knee_local_dir0 = quat_rotate(left_knee_rot, left_knee_local_dir_tile)
    left_knee_local_dir1 = quat_rotate(left_knee_q, left_knee_local_dir_tile)
    left_leg_dot = torch.sum(left_knee_local_dir0 * left_knee_local_dir1, dim=-1)
    left_leg_dot = torch.clamp(left_leg_dot, -1.0, 1.0)
    left_leg_theta = torch.acos(left_leg_dot)
    left_leg_theta = torch.where(left_knee_local_dir0[..., 1] >= 0, left_leg_theta, -left_leg_theta)
    left_leg_q = quat_from_angle_axis(left_leg_theta, left_knee_local_dir.unsqueeze(0))
    left_hip_rot = quat_mul(left_hip_rot, left_leg_q)

    # 更新旋转
    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[..., right_thigh_id, :] = right_hip_rot
    new_local_rotation[..., right_shin_id, :] = right_knee_q
    new_local_rotation[..., left_thigh_id, :] = left_hip_rot
    new_local_rotation[..., left_shin_id, :] = left_knee_q

    # 创建新的骨架状态
    new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation, motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)
    
    return new_motion

def process(i, motion_name, source_tpose, target_tpose, source_motion, target_motion, retarget_data):
    """处理单个动作文件"""
    source_motion_path = os.path.join(source_motion, motion_name + ".npy")
    target_motion_path = os.path.join(target_motion, motion_name + ".npy")
    
    try:
        source_motion = SkeletonMotion.from_file(source_motion_path)
    except:
        print("failed to load motion: ", source_motion_path)
        return
        
    print(f"run retargeting {i}: {motion_name}")
    
    # 执行retarget
    target_motion = source_motion.retarget_to_by_tpose(
      joint_mapping=retarget_data["joint_mapping"],
      source_tpose=source_tpose,
      target_tpose=target_tpose,
      rotation_to_target_skeleton=torch.tensor(retarget_data["rotation"]),
      scale_to_target_skeleton=retarget_data["scale"]
    )
    
    # 处理帧范围
    frame_beg = retarget_data["trim_frame_beg"]
    frame_end = retarget_data["trim_frame_end"]
    if (frame_beg == -1):
        frame_beg = 0
        
    if (frame_end == -1):
        frame_end = target_motion.local_rotation.shape[0]
        
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    local_rotation = local_rotation[frame_beg:frame_end, ...]
    root_translation = root_translation[frame_beg:frame_end, ...]
      
    # 调整高度
    tar_global_pos = target_motion.global_translation[frame_beg:frame_end, ...]
    min_h = torch.min(tar_global_pos[..., 2])
    root_translation[:, 2] += -min_h

    # 创建新的动作
    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # 保存retargeted motion
    target_motion.to_file(target_motion_path)

    return

def save_all():
    """处理所有动作文件"""
    retarget_data_path = "data/configs/retarget_cmu_to_h1_2.json"
    with open(retarget_data_path) as f:
        retarget_data = json.load(f)

    source_tpose = SkeletonState.from_file(retarget_data["source_tpose"])
    target_tpose = SkeletonState.from_file(retarget_data["target_tpose"])
    source_motion = retarget_data["source_motion"]
    target_motion = retarget_data["target_motion_path"]

    # 读取动作列表
    with open("data/configs/motions_autogen_all.yaml", 'r') as f:
        motions_list = yaml.load(f, Loader=yaml.SafeLoader)["motions"]
    
    # 过滤已处理的动作
    all_motion_names = []
    for motion_entry in motions_list.keys():
        if motion_entry == "root":
            continue
        target_motion_file = os.path.join(target_motion, motion_entry + ".npy")
        if os.path.exists(target_motion_file):
            print("Already exists, skip: ", motion_entry)
            continue
        all_motion_names.append(motion_entry)
    all_motion_names.sort()
    
    print(f"Processing {len(all_motion_names)} motions for H1_2...")
    
    # 多进程处理
    n_workers = multiprocessing.cpu_count()
    with multiprocessing.Pool(n_workers) as pool:
        list(tqdm(pool.starmap(process, [
            (i, motion_name, source_tpose, target_tpose, source_motion, target_motion, retarget_data) 
            for i, motion_name in enumerate(all_motion_names)
        ]), total=len(all_motion_names)))
    
if __name__ == '__main__':
    save_all()