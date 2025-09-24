#!/usr/bin/env python3
"""
定义G1机器人的关键关节索引
基于G1的URDF结构和运动学链
"""

import xml.etree.ElementTree as ET

def analyze_g1_urdf():
    """分析G1的URDF结构，提取link和joint信息"""
    urdf_path = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/resources/robots/g1/g1_23dof.urdf"
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # 提取所有links
    links = []
    for link in root.findall('link'):
        link_name = link.get('name')
        if link_name:
            links.append(link_name)
    
    # 提取所有joints
    joints = []
    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        parent = joint.find('parent')
        child = joint.find('child')
        
        if parent is not None and child is not None:
            parent_name = parent.get('link')
            child_name = child.get('link')
            joints.append({
                'name': joint_name,
                'type': joint_type,
                'parent': parent_name,
                'child': child_name
            })
    
    return links, joints

def define_g1_key_bodies():
    """定义G1的关键关节索引"""
    
    # 分析URDF结构
    links, joints = analyze_g1_urdf()
    
    print("=== G1 URDF Analysis ===")
    print(f"Total links: {len(links)}")
    print(f"Total joints: {len(joints)}")
    
    print("\n=== Links ===")
    for i, link in enumerate(links):
        print(f"{i:2d}: {link}")
    
    print("\n=== Joints ===")
    for i, joint in enumerate(joints):
        print(f"{i:2d}: {joint['name']} ({joint['type']}) - {joint['parent']} -> {joint['child']}")
    
    # 定义关键关节（基于G1的URDF结构）
    # G1是12DOF机器人，主要关注下半身
    key_body_names = [
        'pelvis',                    # 0: 骨盆（根节点）
        'left_hip_pitch_link',       # 1: 左髋关节
        'left_knee_link',            # 4: 左膝关节  
        'left_ankle_pitch_link',     # 5: 左踝关节
        'right_hip_pitch_link',      # 7: 右髋关节
        'right_knee_link',           # 10: 右膝关节
        'right_ankle_pitch_link'     # 11: 右踝关节
    ]
    
    print("\n=== Key Body Mapping ===")
    key_body_indices = []
    for body_name in key_body_names:
        if body_name in links:
            idx = links.index(body_name)
            key_body_indices.append(idx)
            print(f"Key body: {body_name} -> index {idx}")
        else:
            print(f"Warning: {body_name} not found in links!")
    
    print(f"\nKey body indices: {key_body_indices}")
    
    # 生成代码
    print("\n=== Generated Code ===")
    print("# G1关键关节索引定义")
    print(f"G1_KEY_BODY_INDICES = {key_body_indices}")
    print(f"G1_KEY_BODY_NAMES = {key_body_names}")
    
    return key_body_indices, key_body_names

if __name__ == "__main__":
    key_indices, key_names = define_g1_key_bodies()