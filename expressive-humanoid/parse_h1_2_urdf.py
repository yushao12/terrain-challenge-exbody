#!/usr/bin/env python3
"""
解析H1_2的URDF文件，提取关节信息用于创建tpose
"""

import xml.etree.ElementTree as ET
import numpy as np

def parse_h1_2_urdf():
    """解析H1_2的URDF文件，提取关节信息"""
    urdf_path = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/resources/robots/h1_2/h1_2_fix_arm.urdf"
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # 存储关节信息
    joints_info = {}
    links_info = {}
    
    # 解析links
    for link in root.findall('link'):
        link_name = link.get('name')
        links_info[link_name] = {'name': link_name}
    
    # 解析joints
    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        
        if joint_type == 'revolute':
            origin = joint.find('origin')
            parent = joint.find('parent')
            child = joint.find('child')
            
            if origin is not None and parent is not None and child is not None:
                xyz = origin.get('xyz', '0 0 0')
                rpy = origin.get('rpy', '0 0 0')
                parent_link = parent.get('link')
                child_link = child.get('link')
                
                xyz_values = [float(x) for x in xyz.split()]
                rpy_values = [float(x) for x in rpy.split()]
                
                joints_info[joint_name] = {
                    'name': joint_name,
                    'type': joint_type,
                    'parent': parent_link,
                    'child': child_link,
                    'origin_xyz': xyz_values,
                    'origin_rpy': rpy_values
                }
    
    # 打印关节信息
    print("H1_2 Joint Information:")
    print("=" * 80)
    
    # 按顺序打印下半身关节
    lower_body_joints = [
        'left_hip_yaw_joint',
        'left_hip_pitch_joint', 
        'left_hip_roll_joint',
        'left_knee_joint',
        'left_ankle_pitch_joint',
        'left_ankle_roll_joint',
        'right_hip_yaw_joint',
        'right_hip_pitch_joint',
        'right_hip_roll_joint', 
        'right_knee_joint',
        'right_ankle_pitch_joint',
        'right_ankle_roll_joint'
    ]
    
    for joint_name in lower_body_joints:
        if joint_name in joints_info:
            joint = joints_info[joint_name]
            print(f"{joint_name}:")
            print(f"  Parent: {joint['parent']}")
            print(f"  Child: {joint['child']}")
            print(f"  Origin XYZ: {joint['origin_xyz']}")
            print(f"  Origin RPY: {joint['origin_rpy']}")
            print()
    
    # 生成骨架结构
    print("H1_2 Skeleton Structure:")
    print("=" * 80)
    
    # 构建父子关系
    joint_to_index = {}
    index = 0
    
    # 添加pelvis作为root
    joint_to_index['pelvis'] = index
    index += 1
    
    # 添加下半身关节
    for joint_name in lower_body_joints:
        if joint_name in joints_info:
            joint_to_index[joint_name] = index
            index += 1
    
    # 打印索引映射
    print("Joint Index Mapping:")
    for joint_name, idx in joint_to_index.items():
        print(f"  {idx:2d}: {joint_name}")
    
    # 生成parent indices
    print("\nParent Indices:")
    parent_indices = []
    for joint_name, idx in joint_to_index.items():
        if joint_name == 'pelvis':
            parent_indices.append(-1)  # root
        else:
            # 找到父关节
            if joint_name in joints_info:
                parent_link = joints_info[joint_name]['parent']
                # 找到父关节的索引
                parent_idx = -1
                for parent_joint, parent_joint_idx in joint_to_index.items():
                    if parent_joint == parent_link:
                        parent_idx = parent_joint_idx
                        break
                parent_indices.append(parent_idx)
            else:
                parent_indices.append(-1)
    
    print("Parent indices array:", parent_indices)
    
    # 生成local translations
    print("\nLocal Translations:")
    local_translations = []
    for joint_name, idx in joint_to_index.items():
        if joint_name == 'pelvis':
            local_translations.append([0.0, 0.0, 0.0])  # root
        else:
            if joint_name in joints_info:
                xyz = joints_info[joint_name]['origin_xyz']
                local_translations.append(xyz)
            else:
                local_translations.append([0.0, 0.0, 0.0])
    
    print("Local translations:")
    for i, trans in enumerate(local_translations):
        joint_name = list(joint_to_index.keys())[i]
        print(f"  {i:2d} ({joint_name:20s}): [{trans[0]:8.3f}, {trans[1]:8.3f}, {trans[2]:8.3f}]")
    
    return joint_to_index, parent_indices, local_translations

if __name__ == "__main__":
    parse_h1_2_urdf()