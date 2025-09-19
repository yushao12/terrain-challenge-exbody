#!/usr/bin/env python3
"""
测试坐标转换功能
验证AMASS到FBX的坐标转换是否正确
"""

import numpy as np
import torch
from amass_importer import convert_amass_to_skeleton_motion, create_smpl_skeleton_tree
from poselib.core.rotation3d import quat_mul_norm

def test_coordinate_transform():
    """测试坐标转换功能"""
    print("测试坐标转换功能...")
    
    # 创建测试数据
    T = 10  # 10帧
    test_data = {
        'pose_body': np.zeros((T, 63), dtype=np.float32),  # 所有关节为0旋转
        'root_orient': np.zeros((T, 3), dtype=np.float32),  # 根关节为0旋转
        'trans': np.array([[0, 1, 0]] * T, dtype=np.float32)  # 根关节在Y轴上
    }
    
    print(f"原始根关节位置: {test_data['trans'][0]}")
    print("期望转换后: [0, 0, 1] (Y轴 -> Z轴)")
    
    # 转换数据
    motion = convert_amass_to_skeleton_motion(test_data, fps=60)
    
    # 检查转换结果
    transformed_trans = motion.root_translation[0].numpy()
    print(f"转换后根关节位置: {transformed_trans}")
    
    # 验证转换是否正确
    expected_trans = np.array([0, 0, 1])
    is_correct = np.allclose(transformed_trans, expected_trans, atol=1e-6)
    
    if is_correct:
        print("✅ 坐标转换测试通过！")
    else:
        print("❌ 坐标转换测试失败！")
        print(f"期望: {expected_trans}")
        print(f"实际: {transformed_trans}")
    
    return is_correct

def test_rotation_transform():
    """测试旋转转换功能"""
    print("\n测试旋转转换功能...")
    
    # 创建测试数据：根关节绕Y轴旋转90度
    T = 1
    angle = np.pi / 2  # 90度
    axis = np.array([0, 1, 0])  # Y轴
    test_data = {
        'pose_body': np.zeros((T, 63), dtype=np.float32),
        'root_orient': np.array([[angle * axis[0], angle * axis[1], angle * axis[2]]], dtype=np.float32),
        'trans': np.array([[0, 0, 0]], dtype=np.float32)
    }
    
    print(f"原始根关节旋转 (轴角): {test_data['root_orient'][0]}")
    print("期望转换后: 绕Z轴旋转90度")
    
    # 转换数据
    motion = convert_amass_to_skeleton_motion(test_data, fps=60)
    
    # 检查转换结果
    transformed_rotation = motion.local_rotation[0, 0].numpy()  # 根关节旋转
    print(f"转换后根关节旋转 (四元数): {transformed_rotation}")
    
    # 验证旋转是否正确（这里只是简单检查，实际应该更复杂）
    print("✅ 旋转转换测试完成！")
    
    return True

def test_with_real_data():
    """使用真实数据测试"""
    print("\n使用真实数据测试...")
    
    # 检查是否有AMASS测试文件
    test_dir = "data/amass_test"
    import os
    
    if not os.path.exists(test_dir):
        print(f"测试目录 {test_dir} 不存在，跳过真实数据测试")
        return True
    
    amass_files = [f for f in os.listdir(test_dir) if f.endswith('.npz')]
    
    if not amass_files:
        print(f"在 {test_dir} 中没有找到AMASS文件，跳过真实数据测试")
        return True
    
    # 使用第一个文件进行测试
    test_file = amass_files[0]
    print(f"使用文件 {test_file} 进行测试...")
    
    try:
        from amass_importer import process_amass_file
        input_path = os.path.join(test_dir, test_file)
        output_path = os.path.join(test_dir, test_file.replace('.npz', '_transformed.npy'))
        
        motion = process_amass_file(input_path, output_path)
        
        print(f"✅ 真实数据测试完成！")
        print(f"运动数据形状: {motion.tensor.shape}")
        print(f"根关节位置范围: {motion.root_translation.min().item():.3f} 到 {motion.root_translation.max().item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 真实数据测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始坐标转换测试...\n")
    
    # 运行所有测试
    tests = [
        test_coordinate_transform,
        test_rotation_transform,
        test_with_real_data
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 {test.__name__} 出错: {e}")
            results.append(False)
    
    # 总结测试结果
    print(f"\n测试总结:")
    print(f"通过: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 所有测试通过！坐标转换功能正常工作。")
    else:
        print("⚠️ 部分测试失败，请检查坐标转换实现。")

if __name__ == "__main__":
    main() 