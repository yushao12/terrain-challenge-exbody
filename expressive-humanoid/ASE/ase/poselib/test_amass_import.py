#!/usr/bin/env python3
"""
测试AMASS导入功能
"""

import os
import numpy as np
import torch
from amass_importer import convert_amass_to_skeleton_motion, create_smpl_skeleton_tree

def create_test_amass_data():
    """创建测试用的AMASS数据"""
    # 创建一个简单的测试动作 (10帧)
    T = 10
    poses = np.zeros((T, 72), dtype=np.float32)  # SMPL pose参数
    trans = np.zeros((T, 3), dtype=np.float32)   # 根位置
    
    # 添加一些简单的动作
    for t in range(T):
        # 简单的行走动作
        poses[t, 0:3] = [0, 0, t * 0.1]  # 根关节旋转
        poses[t, 3:6] = [0, 0, np.sin(t * 0.5) * 0.1]  # 左髋关节
        poses[t, 6:9] = [0, 0, -np.sin(t * 0.5) * 0.1]  # 右髋关节
        
        # 根位置向前移动
        trans[t] = [t * 0.1, 0, 0]
    
    return {
        'poses': poses,
        'trans': trans
    }

def test_smpl_skeleton():
    """测试SMPL骨架创建"""
    print("Testing SMPL skeleton creation...")
    
    skeleton_tree = create_smpl_skeleton_tree()
    print(f"Skeleton tree created with {len(skeleton_tree)} joints")
    print(f"Joint names: {skeleton_tree.node_names[:5]}...")  # 显示前5个关节
    print(f"Parent indices: {skeleton_tree.parent_indices[:5]}...")
    
    return skeleton_tree

def test_amass_conversion():
    """测试AMASS数据转换"""
    print("\nTesting AMASS data conversion...")
    
    # 创建测试数据
    test_data = create_test_amass_data()
    print(f"Test data shapes: poses={test_data['poses'].shape}, trans={test_data['trans'].shape}")
    
    # 转换为SkeletonMotion
    motion = convert_amass_to_skeleton_motion(test_data, fps=30)
    print(f"Motion created: shape={motion.tensor.shape}, fps={motion.fps}")
    
    return motion

def test_save_and_load():
    """测试保存和加载"""
    print("\nTesting save and load...")
    
    # 创建测试数据
    test_data = create_test_amass_data()
    motion = convert_amass_to_skeleton_motion(test_data)
    
    # 保存
    test_file = "data/amass_test/test_motion.npy"
    motion.to_file(test_file)
    print(f"Saved motion to {test_file}")
    
    # 加载
    from poselib.skeleton.skeleton3d import SkeletonMotion
    loaded_motion = SkeletonMotion.from_file(test_file)
    print(f"Loaded motion: shape={loaded_motion.tensor.shape}")
    
    # 验证数据一致性
    assert torch.allclose(motion.tensor, loaded_motion.tensor), "Data mismatch!"
    print("Save and load test passed!")
    
    return loaded_motion

def main():
    """主测试函数"""
    print("=== AMASS Import Test ===")
    
    try:
        # 测试骨架创建
        skeleton_tree = test_smpl_skeleton()
        
        # 测试数据转换
        motion = test_amass_conversion()
        
        # 测试保存和加载
        loaded_motion = test_save_and_load()
        
        print("\n=== All tests passed! ===")
        print("AMASS import functionality is working correctly.")
        
    except Exception as e:
        print(f"\n=== Test failed: {e} ===")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 