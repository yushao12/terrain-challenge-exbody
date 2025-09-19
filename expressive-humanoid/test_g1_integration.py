#!/usr/bin/env python3

"""
Test script to verify G1 robot integration in expressive-humanoid project
"""

import sys
import os

# Add the legged_gym to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym'))

def test_g1_import():
    """Test if G1 robot can be imported successfully"""
    try:
        from legged_gym.envs.g1.g1_config import G1Cfg, G1CfgPPO
        from legged_gym.envs.g1.g1_env import G1Robot
        print("✓ G1 config and environment classes imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import G1 classes: {e}")
        return False

def test_g1_registration():
    """Test if G1 robot is registered in task registry"""
    try:
        from legged_gym.envs import task_registry
        available_tasks = list(task_registry.task_classes.keys())
        print(f"Available tasks: {available_tasks}")
        
        if 'g1' in available_tasks:
            print("✓ G1 robot is registered in task registry")
            return True
        else:
            print("✗ G1 robot is not registered in task registry")
            return False
    except Exception as e:
        print(f"✗ Failed to check task registry: {e}")
        return False

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

def main():
    """Run all tests"""
    print("Testing G1 robot integration in expressive-humanoid project...")
    print("=" * 60)
    
    tests = [
        test_g1_import,
        test_g1_resources,
        test_g1_registration,
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
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)