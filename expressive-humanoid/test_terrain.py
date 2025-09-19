#!/usr/bin/env python3

import sys
import os

# 添加challenging_terrain路径
sys.path.append('/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/challenging_terrain')

try:
    from terrain_base.terrain import Terrain as ChallengingTerrain
    from terrain_base.config import terrain_config
    
    print("="*50)
    print("TESTING TERRAIN INITIALIZATION")
    print("="*50)
    
    # 测试terrain配置
    print(f"Terrain config num_goals: {terrain_config.num_goals}")
    print(f"Terrain config terrain_length: {terrain_config.terrain_length}")
    print(f"Terrain config terrain_width: {terrain_config.terrain_width}")
    print(f"Terrain config num_rows: {terrain_config.num_rows}")
    print(f"Terrain config num_cols: {terrain_config.num_cols}")
    
    # 测试terrain初始化
    print("\nInitializing terrain...")
    terrain = ChallengingTerrain(num_robots=1)
    
    print(f"Terrain type: {type(terrain)}")
    print(f"Terrain goals shape: {terrain.goals.shape if hasattr(terrain, 'goals') else 'No goals'}")
    print(f"Terrain num_goals: {terrain.num_goals if hasattr(terrain, 'num_goals') else 'No num_goals'}")
    
    if hasattr(terrain, 'goals'):
        print(f"Terrain goals: {terrain.goals}")
        print(f"Goals for terrain[0,0]: {terrain.goals[0,0] if terrain.goals.shape[0] > 0 and terrain.goals.shape[1] > 0 else 'No goals'}")
    
    print("\nTerrain initialization successful!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 