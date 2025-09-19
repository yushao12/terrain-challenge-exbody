# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from isaacgym.torch_utils import *
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

"""
修改版retarget脚本，跳过交互式可视化，保存图片序列
"""

VISUALIZE = False

def save_motion_frames(motion, output_dir="motion_frames", skip_frames=10):
    """
    保存动作序列为图片帧
    
    Args:
        motion: SkeletonMotion对象
        output_dir: 输出目录
        skip_frames: 跳过的帧数（每skip_frames帧保存一张）
    """
    print(f"保存动作帧到 {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置matplotlib后端为Agg（无显示）
    plt.switch_backend('Agg')
    
    # 获取全局位置
    global_pos = motion.global_translation
    
    # 计算边界框
    all_positions = global_pos.reshape(-1, 3)
    min_pos = torch.min(all_positions, dim=0)[0]
    max_pos = torch.max(all_positions, dim=0)[0]
    center = (min_pos + max_pos) / 2
    size = max_pos - min_pos
    max_size = torch.max(size)
    
    # 保存帧
    for i in range(0, len(motion), skip_frames):
        print(f"保存帧 {i}/{len(motion)}")
        
        # 创建图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制骨架
        frame_pos = global_pos[i]  # [num_joints, 3]
        
        # 绘制关节
        ax.scatter(frame_pos[:, 0], frame_pos[:, 1], frame_pos[:, 2], 
                  c='red', s=50, alpha=0.8)
        
        # 绘制连接线（简化版本）
        # 这里可以根据skeleton_tree的parent_indices绘制连接线
        # 暂时跳过，只显示关节点
        
        # 设置坐标轴
        ax.set_xlim(center[0] - max_size/2, center[0] + max_size/2)
        ax.set_ylim(center[1] - max_size/2, center[1] + max_size/2)
        ax.set_zlim(center[2] - max_size/2, center[2] + max_size/2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {i}')
        
        # 保存图片
        plt.savefig(os.path.join(output_dir, f'frame_{i:04d}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"保存完成！共保存 {len(range(0, len(motion), skip_frames))} 帧")

def main():
    # load retarget config
    retarget_data_path = "data/configs/retarget_smpl_to_h1.json"
    with open(retarget_data_path) as f:
        retarget_data = json.load(f)
    
    print("加载T-pose文件...")
    # load t-pose files
    source_tpose = SkeletonState.from_file(retarget_data["source_tpose"])
    target_tpose = SkeletonState.from_file(retarget_data["target_tpose"])

    print("加载源动作文件...")
    # load source motion sequence
    source_motion = SkeletonMotion.from_file(retarget_data["source_motion"])
    print(f"源动作: {len(source_motion)} 帧")

    # parse data from retarget config
    joint_mapping = retarget_data["joint_mapping"]
    rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])

    print("开始retarget...")
    # run retargeting
    target_motion = source_motion.retarget_to_by_tpose(
      joint_mapping=retarget_data["joint_mapping"],
      source_tpose=source_tpose,
      target_tpose=target_tpose,
      rotation_to_target_skeleton=rotation_to_target_skeleton,
      scale_to_target_skeleton=retarget_data["scale"]
    )
    print(f"Retarget完成: {len(target_motion)} 帧")

    # keep frames between [trim_frame_beg, trim_frame_end - 1]
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

    # move the root so that the feet are on the ground
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    tar_global_pos = target_motion.global_translation
    min_h = torch.min(tar_global_pos[..., 2])
    root_translation[:, 2] += -min_h
    
    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    print("保存retargeted动作文件...")
    # save retargeted motion
    target_motion.to_file(retarget_data["target_motion_path"])
    print(f"保存到: {retarget_data['target_motion_path']}")

    print("保存动作帧图片...")
    # save motion frames as images
    save_motion_frames(target_motion, "motion_frames", skip_frames=20)
    
    print("完成！")
    return

if __name__ == '__main__':
    main() 