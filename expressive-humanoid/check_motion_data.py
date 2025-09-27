#!/usr/bin/env python3

import numpy as np
import os
from pathlib import Path

def check_motion_data():
    """检查motion数据的格式"""
    
    motion_dir = Path("/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/ASE/ase/data/motions")
    
    # 检查几个不同的motion文件
    test_files = [
        "amp_humanoid_walk.npy",
        "amp_humanoid_run.npy", 
        "amp_humanoid_jog.npy"
    ]
    
    for filename in test_files:
        filepath = motion_dir / filename
        if filepath.exists():
            print(f"\n检查文件: {filename}")
            print("=" * 50)
            
            try:
                # 加载数据
                data = np.load(filepath, allow_pickle=True)
                print(f"数据类型: {type(data)}")
                print(f"数据形状: {data.shape}")
                print(f"数据dtype: {data.dtype}")
                
                # 如果是object类型，尝试访问内容
                if data.dtype == object:
                    if hasattr(data, 'item'):
                        item = data.item()
                        print(f"Item类型: {type(item)}")
                        if isinstance(item, dict):
                            print(f"字典键: {list(item.keys())}")
                            for key, value in item.items():
                                if hasattr(value, 'shape'):
                                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                                else:
                                    print(f"  {key}: {type(value)}")
                        elif hasattr(item, 'shape'):
                            print(f"Item形状: {item.shape}")
                            print(f"Item dtype: {item.dtype}")
                
                # 尝试直接访问数据
                if hasattr(data, 'shape') and len(data.shape) > 0:
                    print(f"数据范围: [{data.min():.3f}, {data.max():.3f}]")
                    
            except Exception as e:
                print(f"错误: {e}")
    
    # 检查reallusion数据
    reallusion_dir = motion_dir / "reallusion_sword_shield"
    if reallusion_dir.exists():
        print(f"\n检查Reallusion数据目录")
        print("=" * 50)
        
        npy_files = list(reallusion_dir.glob("*.npy"))
        if npy_files:
            test_file = npy_files[0]
            print(f"测试文件: {test_file.name}")
            
            try:
                data = np.load(test_file, allow_pickle=True)
                print(f"数据类型: {type(data)}")
                print(f"数据形状: {data.shape}")
                print(f"数据dtype: {data.dtype}")
                
                if data.dtype == object:
                    if hasattr(data, 'item'):
                        item = data.item()
                        print(f"Item类型: {type(item)}")
                        if isinstance(item, dict):
                            print(f"字典键: {list(item.keys())}")
                
            except Exception as e:
                print(f"错误: {e}")

if __name__ == "__main__":
    check_motion_data()