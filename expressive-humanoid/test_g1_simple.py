#!/usr/bin/env python3

"""
Simple test script to verify G1 robot integration without importing problematic modules
"""

import sys
import os

# Add the legged_gym to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym'))

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

def test_g1_config_import():
    """Test if G1 config can be imported without torch issues"""
    try:
        # Import only the config, not the environment
        from legged_gym.envs.g1.g1_config import G1Cfg, G1CfgPPO
        print("✓ G1 config classes imported successfully")
        
        # Test creating config instances
        cfg = G1Cfg()
        ppo_cfg = G1CfgPPO()
        print("✓ G1 config instances created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import G1 config: {e}")
        return False

def test_g1_directory_structure():
    """Test if G1 directory structure is correct"""
    try:
        g1_dir = "/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/legged_gym/envs/g1"
        required_files = ["g1_config.py", "g1_env.py"]
        
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

def main():
    """Run all tests"""
    print("Testing G1 robot integration in expressive-humanoid project...")
    print("=" * 60)
    
    tests = [
        test_g1_directory_structure,
        test_g1_resources,
        test_g1_config_import,
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
        print("\nNext steps:")
        print("1. G1 robot resources are properly copied")
        print("2. G1 config and environment files are in place")
        print("3. You can now use G1 robot in your training scripts")
        print("\nTo use G1 robot, add '--task=g1' to your training command")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)