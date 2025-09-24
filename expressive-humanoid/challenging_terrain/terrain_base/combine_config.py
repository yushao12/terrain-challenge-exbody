from .single_terrain import single_terrain
from .config import terrain_config
import numpy as np
class combine_config:
        single = [
                single_terrain.parkour, #0
                single_terrain.parkour_training_stage1, #1 - 新增：parkour训练阶段1变体
                single_terrain.hurdle,#2
                single_terrain.bridge,#3
                single_terrain.flat,#4
                single_terrain.uneven,#5
                single_terrain.stair,#6
                single_terrain.wave,#7
                single_terrain.slope,#8
                single_terrain.gap,#9
        ]

        multiplication = [
                [single[9],single[3],],#gap_bridge,0 (更新索引)
                [single[7],single[9]],#wave_gap,1 (更新索引)
                [single[7],single[3]],#wave_bridge,2 (更新索引)
        ]

        addition = [
                [single[6],single[3],single[5],single[9]] # 更新索引
        ]

        proportions = [
                # 第一阶段训练：使用parkour_training_stage1 (索引1)
              #  ("single", 1, 1),  # parkour_training_stage1
                
                # 其他地形配置（注释掉，专注于parkour训练）
                #("single", 3, 1),
                ("single", 1, 0.1),
                ("single", 2, 0.1),
                ("single", 3, 0.1),
                ("single", 4, 0.1),
                ("single", 6, 0.1),
                ("single", 7, 0.1),
               # ("single", 7, 1),  # wave地形
              # ("single", 7, 0.3),   # slope地形
              # ("multiplication", 2, 1)
              # ("multiplication", 0, 0.05),
               ("addition", 0, 0.05),
               ("single", 5, 0.1),
              # ("single", 3, 1),
              # ("multiplication", 1, 0.05),       
              # ("multiplication", 2, 0.05),       
        ]

class generator:
        def __init__(self, cfg: terrain_config) -> None:
                self.cfg = cfg

        def single_create(terrain,id=0,difficulty=0.5):
                length_x = terrain_config.terrain_length
                length_y = terrain_config.terrain_width
                num_goals = terrain_config.num_goals
                horizontal_scale = terrain.horizontal_scale
                platform_size = terrain_config.platform_size
                terrain , goals, final_x= combine_config.single[id](
                                                                terrain=terrain, 
                                                                length_x=length_x, 
                                                                length_y=length_y, 
                                                                num_goals=num_goals, 
                                                                platform_size=platform_size, 
                                                                difficulty=difficulty)
                terrain.goals = goals * horizontal_scale
                terrain.idx = id
                return terrain

        def addition_create(terrain,id=0,difficulty=0.5):
                terrain_list = combine_config.addition[id]
                num_terrain = len(terrain_list)
                platform_size = terrain_config.platform_size
                length_x = (terrain_config.terrain_length) // num_terrain
                length_y = terrain_config.terrain_width
                num_goals = terrain_config.num_goals // num_terrain
                horizontal_scale = terrain.horizontal_scale
                goals = []
                final_x = 20
                for i in range(num_terrain):
                        if(i == num_terrain-1):
                                num_goals = terrain_config.num_goals - i*num_goals
                        terrain , sub_goals, final_x = terrain_list[i](
                                                                        terrain=terrain, 
                                                                        length_x=length_x, 
                                                                        length_y=length_y, 
                                                                        num_goals=num_goals, 
                                                                        start_x=final_x, 
                                                                        platform_size=platform_size, 
                                                                        difficulty=difficulty)
                        goals.append(sub_goals)

                goals = np.vstack(goals)
                terrain.goals = goals * horizontal_scale
                terrain.idx = id + len(combine_config.single)
                return terrain

        def multiplication_create(terrain,id=0,difficulty=0.5):

                terrain_list = combine_config.multiplication[id]
                num_terrain = len(terrain_list)
                platform_size = terrain_config.platform_size
                length_x = terrain_config.terrain_length- platform_size
                length_y = terrain_config.terrain_width
                num_goals = terrain_config.num_goals
                horizontal_scale = terrain.horizontal_scale
                goals = []
                start_x = 20
                for i in range(num_terrain):
                        terrain , goals, final_x = terrain_list[i](
                                                                terrain=terrain, 
                                                                length_x=length_x, 
                                                                length_y=length_y, 
                                                                num_goals=num_goals, 
                                                                start_x=start_x, 
                                                                platform_size=platform_size, 
                                                                difficulty=difficulty)

                terrain.goals = goals * horizontal_scale
                terrain.idx = id+ len(combine_config.single) + len(combine_config.addition)
                return terrain
