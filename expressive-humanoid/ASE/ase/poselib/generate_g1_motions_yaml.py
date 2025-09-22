#!/usr/bin/env python3
"""
生成G1的motion yaml配置文件
"""

import os
import yaml

def generate_g1_motions_yaml():
    """生成G1的motion yaml文件"""
    
    # G1重定向数据目录
    g1_motion_dir = "data/retarget_npy_g1"
    
    # 获取所有npy文件名
    motion_files = []
    for file in os.listdir(g1_motion_dir):
        if file.endswith('.npy'):
            motion_name = file[:-4]  # 去掉.npy后缀
            motion_files.append(motion_name)
    
    # 按名称排序
    motion_files.sort()
    
    print(f"Found {len(motion_files)} G1 motion files")
    
    # 构建motions字典
    motions = {}
    for motion_name in motion_files:
        motions[motion_name] = {
            'description': 'g1_motion',
            'difficulty': 4,  # 默认难度
            'trim_beg': -1,
            'trim_end': -1,
            'weight': 1.0
        }
    
    # 创建完整的yaml内容
    yaml_content = {
        'motions': motions
    }
    
    # 保存到文件
    output_path = "data/configs/motions_g1_all.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=True)
    
    print(f"Generated {output_path} with {len(motions)} motions")
    
    return output_path

if __name__ == "__main__":
    generate_g1_motions_yaml()