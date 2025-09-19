#!/usr/bin/env python3
"""
详细分析AMASS数据格式
"""

import numpy as np

def analyze_amass_data():
    """分析AMASS数据格式"""
    print("=== AMASS数据格式详细分析 ===")
    
    # 加载AMASS数据
    data_path = "data/amass_test/B1_-_stand_to_walk_stageii.npz"
    data = np.load(data_path, allow_pickle=True)
    
    print(f"文件: {data_path}")
    print(f"所有字段: {list(data.keys())}")
    print()
    
    # 分析每个字段
    for key in data.keys():
        value = data[key]
        print(f"字段: {key}")
        print(f"  类型: {type(value)}")
        print(f"  形状: {value.shape}")
        print(f"  数据类型: {value.dtype}")
        
        if hasattr(value, 'dtype') and value.dtype == np.float64:
            if len(value.shape) == 1:
                print(f"  数值范围: [{value.min():.3f}, {value.max():.3f}]")
            elif len(value.shape) == 2:
                print(f"  数值范围: [{value.min():.3f}, {value.max():.3f}]")
                print(f"  前5个值: {value[0, :5] if value.shape[1] >= 5 else value[0]}")
        print()
    
    # 特别分析pose相关字段
    print("=== Pose字段详细分析 ===")
    
    if 'pose_body' in data:
        pose_body = data['pose_body']
        print(f"pose_body: {pose_body.shape}")
        print(f"  这应该是SMPL的body pose参数")
        print(f"  如果是SMPL格式，应该是 (T, 69) - 23个关节 × 3维轴角")
        print(f"  如果是SMPL-X格式，可能是 (T, 63) - 21个关节 × 3维轴角")
        print(f"  实际形状: {pose_body.shape}")
        
        if pose_body.shape[1] == 69:
            print("  ✓ 这看起来是SMPL格式的body pose")
        elif pose_body.shape[1] == 63:
            print("  ✓ 这看起来是SMPL-X格式的body pose")
        else:
            print(f"  ? 未知格式，维度: {pose_body.shape[1]}")
    
    if 'root_orient' in data:
        root_orient = data['root_orient']
        print(f"root_orient: {root_orient.shape}")
        print(f"  这应该是根关节的方向 (通常是3维轴角)")
    
    if 'pose_hand' in data:
        pose_hand = data['pose_hand']
        print(f"pose_hand: {pose_hand.shape}")
        print(f"  手部pose参数")
    
    if 'pose_jaw' in data:
        pose_jaw = data['pose_jaw']
        print(f"pose_jaw: {pose_jaw.shape}")
        print(f"  下巴pose参数")
    
    if 'pose_eye' in data:
        pose_eye = data['pose_eye']
        print(f"pose_eye: {pose_eye.shape}")
        print(f"  眼部pose参数")
    
    # 验证pose_body + root_orient是否等于poses
    if 'pose_body' in data and 'root_orient' in data and 'poses' in data:
        pose_body = data['pose_body']
        root_orient = data['root_orient']
        poses = data['poses']
        
        print(f"\n=== Pose组合验证 ===")
        print(f"pose_body: {pose_body.shape}")
        print(f"root_orient: {root_orient.shape}")
        print(f"poses: {poses.shape}")
        
        # 检查pose_body + root_orient是否等于poses
        combined_dim = pose_body.shape[1] + root_orient.shape[1]
        print(f"pose_body + root_orient 维度: {combined_dim}")
        print(f"poses 维度: {poses.shape[1]}")
        
        if combined_dim == poses.shape[1]:
            print("✓ poses = pose_body + root_orient")
        else:
            print("✗ poses ≠ pose_body + root_orient")
            
        # 检查pose_hand等是否包含在poses中
        total_pose_dim = pose_body.shape[1] + root_orient.shape[1]
        if 'pose_hand' in data:
            total_pose_dim += data['pose_hand'].shape[1]
        if 'pose_jaw' in data:
            total_pose_dim += data['pose_jaw'].shape[1]
        if 'pose_eye' in data:
            total_pose_dim += data['pose_eye'].shape[1]
            
        print(f"所有pose字段总维度: {total_pose_dim}")
        if total_pose_dim == poses.shape[1]:
            print("✓ poses包含所有pose字段")
        else:
            print("✗ poses不包含所有pose字段")

def check_smpl_vs_smplx():
    """检查是SMPL还是SMPL-X格式"""
    print("\n=== SMPL vs SMPL-X 格式判断 ===")
    
    data_path = "data/amass_test/B1_-_stand_to_walk_stageii.npz"
    data = np.load(data_path, allow_pickle=True)
    
    if 'pose_body' in data:
        pose_body_dim = data['pose_body'].shape[1]
        
        if pose_body_dim == 69:
            print("✓ 这是SMPL格式")
            print("  - 23个body关节 × 3维轴角 = 69")
            print("  - 加上root_orient(3) = 72")
        elif pose_body_dim == 63:
            print("✓ 这是SMPL-X格式")
            print("  - 21个body关节 × 3维轴角 = 63")
            print("  - 加上root_orient(3) = 66")
        else:
            print(f"? 未知格式，pose_body维度: {pose_body_dim}")
    
    # 检查betas维度
    if 'betas' in data:
        betas_dim = data['betas'].shape[0]
        if betas_dim == 10:
            print("✓ 使用10个SMPL体型参数")
        elif betas_dim == 16:
            print("✓ 使用16个SMPL-X体型参数")
        else:
            print(f"? 未知betas维度: {betas_dim}")

def main():
    """主函数"""
    analyze_amass_data()
    check_smpl_vs_smplx()
    
    print("\n=== 分析完成 ===")
    print("根据分析结果，我们需要修改amass_importer.py来正确处理AMASS数据格式")

if __name__ == "__main__":
    main() 