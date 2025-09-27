#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import numpy as np

def parse_h1_2_urdf():
    """解析H1_2的URDF文件，获取link信息"""
    urdf_path = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/resources/robots/h1_2/h1_2_fix_arm.urdf"
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    links = []
    for i, link in enumerate(root.findall('link')):
        link_name = link.get('name')
        links.append((i, link_name))
    
    print("H1_2 URDF Link Information:")
    print("Index | Link Name")
    print("-" * 50)
    for idx, name in links:
        print(f"{idx:5d} | {name}")
    
    return links

def suggest_h1_2_key_bodies():
    """为H1_2建议关键关节"""
    links = parse_h1_2_urdf()
    
    # 基于H1_2的URDF结构，建议关键关节
    # H1_2和G1有相似的结构，都是12个DOF的人形机器人
    suggested_key_bodies = {
        'pelvis': None,
        'left_hip_pitch_link': None,
        'left_knee_link': None,
        'left_ankle_pitch_link': None,
        'right_hip_pitch_link': None,
        'right_knee_link': None,
        'right_ankle_pitch_link': None,
    }
    
    print("\nSuggested H1_2 Key Bodies:")
    print("=" * 50)
    
    for key_name in suggested_key_bodies.keys():
        for idx, link_name in links:
            if key_name in link_name:
                suggested_key_bodies[key_name] = idx
                print(f"{key_name:25s} -> Index {idx:2d} ({link_name})")
                break
    
    # 生成key body indices数组
    key_body_indices = []
    for key_name, idx in suggested_key_bodies.items():
        if idx is not None:
            key_body_indices.append(idx)
    
    print(f"\nKey Body Indices Array: {key_body_indices}")
    print(f"Number of key bodies: {len(key_body_indices)}")
    
    return key_body_indices

if __name__ == "__main__":
    suggest_h1_2_key_bodies()