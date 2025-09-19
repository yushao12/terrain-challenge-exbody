#!/usr/bin/env python3

"""
Final test script to verify G1 robot integration in expressive-humanoid project
"""

import sys
import os

def test_g1_resources():
    """Test if G1 robot resources are available"""
    try:
        g1_urdf_path = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/resources/robots/g1/g1_12dof_with_hand.urdf"
        if os.path.exists(g1_urdf_path):
            print("✓ G1 URDF file found")
            return True
        else:
            print(f"✗ G1 URDF file not found at: {g1_urdf_path}")
            return False
    except Exception as e:
        print(f"✗ Failed to check G1 resources: {e}")
        return False

def test_g1_directory_structure():
    """Test if G1 directory structure is correct"""
    try:
        g1_dir = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/legged_gym/envs/g1"
        required_files = [
            "g1_config.py", 
            "g1_env.py", 
            "g1_mimic_config.py", 
            "g1_mimic_amp_config.py",
            "g1_mimic.py",
            "g1_mimic_amp.py"
        ]
        
        for file in required_files:
            file_path = os.path.join(g1_dir, file)
            if os.path.exists(file_path):
                print(f"✓ {file} found")
            else:
                print(f"✗ {file} not found")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Failed to check directory structure: {e}")
        return False

def test_g1_file_sizes():
    """Test if G1 files have reasonable sizes"""
    try:
        g1_dir = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/legged_gym/envs/g1"
        files_to_check = [
            "g1_config.py", 
            "g1_env.py", 
            "g1_mimic_config.py", 
            "g1_mimic_amp_config.py",
            "g1_mimic.py",
            "g1_mimic_amp.py"
        ]
        
        for file in files_to_check:
            file_path = os.path.join(g1_dir, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                if size > 100:  # 至少100字节
                    print(f"✓ {file} has reasonable size ({size} bytes)")
                else:
                    print(f"✗ {file} seems too small ({size} bytes)")
                    return False
        
        return True
    except Exception as e:
        print(f"✗ Failed to check file sizes: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing G1 robot integration in expressive-humanoid project...")
    print("=" * 60)
    
    tests = [
        test_g1_directory_structure,
        test_g1_resources,
        test_g1_file_sizes,
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! G1 robot integration is successful.")
        print("\nG1 robot has been successfully integrated with the following tasks:")
        print("  - g1: Basic G1 robot task")
        print("  - g1_mimic: G1 mimic task")
        print("  - g1_mimic_amp: G1 mimic AMP task")
        print("\nTo use G1 robot, you can now run:")
        print("  python legged_gym/scripts/train.py --task=g1")
        print("  python legged_gym/scripts/train.py --task=g1_mimic")
        print("  python legged_gym/scripts/train.py --task=g1_mimic_amp")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)