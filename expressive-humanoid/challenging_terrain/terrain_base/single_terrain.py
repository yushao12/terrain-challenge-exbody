import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from .config import terrain_config
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation
import math

class single_terrain:
    def __init__(self, cfg: terrain_config) -> None:
        self.cfg = cfg
    
    def parkour(terrain, 
            length_x=18.,
            length_y=4.,
            num_goals=6, 
            start_x=0,
            start_y=0,
            platform_size=2.5, 
            difficulty=0.5,
            x_range=[0.5, 1.0],
            y_range=[0.3, 0.4],
            stone_len_range=[0.8, 1.0],
            stone_width_range=[0.6, 0.8],
            incline_height=0.1,
            pit_depth=[0.5, 1.]):
    
        goals = np.zeros((num_goals, 2))
        pit_depth_val = np.random.uniform(pit_depth[0], pit_depth[1])
        pit_depth_grid = -round(pit_depth_val / terrain.vertical_scale)
        
        h_scale = terrain.horizontal_scale
        v_scale = terrain.vertical_scale
    
        length_y_grid = round(length_y / h_scale)
        mid_y = length_y_grid // 2

        length_x_grid = round(length_x / h_scale)
        
        stone_len = round(((stone_len_range[0] - stone_len_range[1]) * difficulty + stone_len_range[1]) / h_scale)
        stone_width = round(((stone_width_range[0] - stone_width_range[1]) * difficulty + stone_width_range[1]) / h_scale)
        gap_x = round(((x_range[1] - x_range[0]) * difficulty + x_range[0]) / h_scale)
        gap_y = round(((y_range[1] - y_range[0]) * difficulty + y_range[0]) / h_scale)
        
        platform_size_grid = int(round(platform_size / h_scale))
        incline_height_grid = int(round(incline_height / v_scale))
        
        terrain.height_field_raw[start_x+platform_size_grid:start_x + length_x_grid, start_y:start_y+length_y_grid*2] = pit_depth_grid
        
        dis_x = start_x +platform_size_grid - gap_x + stone_len // 2
        goals[0] = [start_x + platform_size_grid - stone_len // 2, start_y + mid_y]
        left_right_flag = np.random.randint(0, 2)
        
        for i in range(num_goals - 2):
            dis_x += gap_x
            pos_neg = 2 * (left_right_flag - 0.5)  # 1 或 -1
            dis_y = mid_y + pos_neg * gap_y
            
            x_start = int(dis_x - stone_len // 2)
            x_end = x_start + stone_len
            y_start = int(dis_y - stone_width // 2)
            y_end = y_start + stone_width
            
            heights = np.tile(np.linspace(-incline_height_grid, incline_height_grid, stone_width),(stone_len, 1)) * pos_neg
            heights = heights.astype(int)
            
            if x_end > terrain.height_field_raw.shape[0]:
                x_end = terrain.height_field_raw.shape[0]
            if y_end > terrain.height_field_raw.shape[1]:
                y_end = terrain.height_field_raw.shape[1]
    
            actual_height = heights[:x_end - x_start, :y_end - y_start]
            terrain.height_field_raw[x_start:x_end, y_start:y_end] = actual_height
            goals[i + 1] = [dis_x, dis_y]
            left_right_flag = 1 - left_right_flag
        
        final_dis_x = dis_x + gap_x
        goals[-1] = [final_dis_x, mid_y]

        # terrain.height_field_raw[final_dis_x:round(length_x/terrain.horizontal_scale), start_y:start_y+mid_y*2] = 0
        return terrain, goals, final_dis_x
    
    def parkour_training_stage1(terrain, 
            length_x=18.,
            length_y=4.,
            num_goals=6, 
            start_x=0,
            start_y=0,
            platform_size=2.5, 
            difficulty=0.5,
            x_range=[0.5, 1.0],
            y_range=[0.3, 0.4],
            stone_len_range=[0.8, 1.0],
            stone_width_range=[0.6, 0.8],
            incline_height=0.1,
            pit_depth=[0.5, 1.]):
        """
        Parkour地形变体：用于两阶段训练的第一阶段
        - 物理地形：将深坑填充为平地，便于学习基本步态
        - Scan信息：保留原始parkour地形，用于正确的奖励计算
        """
        
        # 首先生成原始parkour地形作为scan参考
        original_terrain = terrain_utils.SubTerrain(
            "terrain",
            width=terrain.width,
            length=terrain.length,
            vertical_scale=terrain.vertical_scale,
            horizontal_scale=terrain.horizontal_scale
        )
        original_terrain.height_field_raw = terrain.height_field_raw.copy()
        
        # 调用原始parkour函数生成地形
        original_terrain, goals, final_dis_x = single_terrain.parkour(
            original_terrain, length_x, length_y, num_goals, start_x, start_y,
            platform_size, difficulty, x_range, y_range, stone_len_range,
            stone_width_range, incline_height, pit_depth
        )
        
        # 保存原始地形作为scan参考
        terrain.scan_reference = original_terrain.height_field_raw.copy()
        terrain.valid_standing_mask = np.zeros_like(terrain.height_field_raw)
        
        # 创建可站立区域掩码
        h_scale = terrain.horizontal_scale
        v_scale = terrain.vertical_scale
        
        length_y_grid = round(length_y / h_scale)
        mid_y = length_y_grid // 2
        length_x_grid = round(length_x / h_scale)
        
        stone_len = round(((stone_len_range[0] - stone_len_range[1]) * difficulty + stone_len_range[1]) / h_scale)
        stone_width = round(((stone_width_range[0] - stone_width_range[1]) * difficulty + stone_width_range[1]) / h_scale)
        gap_x = round(((x_range[1] - x_range[0]) * difficulty + x_range[0]) / h_scale)
        
        platform_size_grid = int(round(platform_size / h_scale))
        
        # 标记起始平台为可站立区域
        terrain.valid_standing_mask[start_x:start_x+platform_size_grid, start_y:start_y+length_y_grid] = 1
        
        # 根据goals位置标记石头区域为可站立
        dis_x = start_x + platform_size_grid - gap_x + stone_len // 2
        left_right_flag = np.random.randint(0, 2)
        
        for i in range(num_goals - 2):
            dis_x += gap_x
            pos_neg = 2 * (left_right_flag - 0.5)
            dis_y = mid_y + pos_neg * round(((y_range[1] - y_range[0]) * difficulty + y_range[0]) / h_scale)
            
            x_start = int(dis_x - stone_len // 2)
            x_end = x_start + stone_len
            y_start = int(dis_y - stone_width // 2)
            y_end = y_start + stone_width
            
            # 确保坐标在有效范围内
            x_start = max(0, x_start)
            x_end = min(terrain.height_field_raw.shape[0], x_end)
            y_start = max(0, y_start)
            y_end = min(terrain.height_field_raw.shape[1], y_end)
            
            # 标记石头区域为可站立
            terrain.valid_standing_mask[x_start:x_end, y_start:y_end] = 1
            left_right_flag = 1 - left_right_flag
        
        # 标记终点平台为可站立区域
        final_dis_x = dis_x + gap_x
        final_x_start = max(0, int(final_dis_x - stone_len // 2))
        final_x_end = min(terrain.height_field_raw.shape[0], int(final_dis_x + stone_len // 2))
        final_y_start = max(0, int(mid_y - stone_width // 2))
        final_y_end = min(terrain.height_field_raw.shape[1], int(mid_y + stone_width // 2))
        terrain.valid_standing_mask[final_x_start:final_x_end, final_y_start:final_y_end] = 1
        
        # 阶段1：保留原始parkour石头的完整设计，只将非石头区域填成平地
        # 这样机器人可以在平坦"地面"上学习parkour步态，同时体验真实石头的斜度
        
        # 首先复制原始parkour地形（包括石头的斜度）
        terrain.height_field_raw = original_terrain.height_field_raw.copy()
        
        # 将非可站立区域（坑和空隙）填成平地
        non_standing_mask = terrain.valid_standing_mask == 0
        terrain.height_field_raw[non_standing_mask] = 0
        
        # 设置训练阶段标识
        terrain.training_stage = 1
        
        return terrain, goals, final_dis_x
    
    def hurdle(
            terrain,
            length_x=18.,
            length_y=4.,
            num_goals=8,
            start_x=0,
            start_y=0,
            platform_size=1., 
            difficulty = 0.5,
            hurdle_range=[0.1, 0.2],
            hurdle_height_range=[0.1, 0.2],
            flat_size = 0.6
            ):
        
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale)// 2  
        per_x = (round(length_x/ terrain.horizontal_scale)- platform_size) // num_goals


        hurdle_size = round(((hurdle_range[1]-hurdle_range[0])*difficulty +hurdle_range[0])/terrain.horizontal_scale)
        hurdle_height = round(((hurdle_height_range[1]-hurdle_height_range[0])*difficulty + hurdle_height_range[0])/terrain.vertical_scale)

        platform_size = round(platform_size / terrain.horizontal_scale)
        # terrain.height_field_raw[start_x:start_x+platform_size, start_y:start_y+2*mid_y] = 0

        terrain.height_field_raw[start_x:start_x +round(length_x/ terrain.horizontal_scale), start_y:start_y+mid_y*2] = 0

        flat_size = round(flat_size / terrain.horizontal_scale)
        dis_x = start_x + platform_size

        for i in range(num_goals):
            goals[i]=[dis_x+per_x*i,start_y+mid_y]

        for i in range(num_goals):

            terrain.height_field_raw[dis_x-hurdle_size//2:dis_x+hurdle_size//2, start_y:start_y+mid_y*2] = hurdle_height
            dis_x += flat_size + hurdle_size

        return terrain,goals,dis_x
        
    def bridge(terrain,
               length_x=18.0,
                length_y=4.0,
                num_goals=8,
                start_x = 0,
                start_y = 0,
                platform_size=1.0, 
                difficulty = 0.5,
                bridge_width_range=[0.3,0.4],  
                bridge_height=0.7,
                ):
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y / terrain.horizontal_scale) // 2  
        bridge_width = round(((bridge_width_range[1]-bridge_width_range[0])*difficulty +bridge_width_range[0])/terrain.horizontal_scale)
        bridge_height = round(bridge_height / terrain.vertical_scale)
        platform_size = round(platform_size / terrain.horizontal_scale)
        terrain.height_field_raw[start_x:start_x+platform_size, start_y:start_y+2*mid_y] = 0
        bridge_start_x = platform_size + start_x
        bridge_length = round(length_x / terrain.horizontal_scale)
        bridge_end_x = start_x + bridge_length

        for i in range(num_goals):
            goals[i] = [bridge_start_x + bridge_length/num_goals*i, mid_y]  
       
        left_y1 = 0
        left_y2 = int(mid_y - bridge_width // 2) 
        right_y1 = int(mid_y + bridge_width // 2)
        right_y2 = mid_y*2
        terrain.height_field_raw[bridge_start_x:bridge_end_x, left_y1:left_y2] = -bridge_height
        terrain.height_field_raw[bridge_start_x:bridge_end_x, right_y1:right_y2] = -bridge_height

        # terrain.height_field_raw[bridge_start_x:bridge_end_x, left_y2:right_y1] = 0

        return terrain,goals,bridge_end_x

    def flat(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            ):
        goals = np.zeros((num_goals, 2))
        length_x = round(length_x / terrain.horizontal_scale)
        length_y = round(length_y / terrain.horizontal_scale)
        platform_size = round(platform_size / terrain.horizontal_scale)

        for i in range(num_goals):
            # y_pos = round(random.uniform(0,length_y))
            y_pos = length_y//2
            goals[i]=[start_x+platform_size+length_x/num_goals*i,start_y+y_pos]

        return terrain,goals,length_x

    def uneven(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            num_range=[150,200],
            size_range=[0.4,0.7],
            height_range=[0.1,0.2],
            ):   

        goals = np.zeros((num_goals, 2))
        platform_size = round(platform_size/ terrain.horizontal_scale)
        per_x = (round(length_x/ terrain.horizontal_scale) - platform_size)// num_goals
        mid_y = round(length_y/ terrain.horizontal_scale) // 2

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+per_x*i,start_y+mid_y]

        height = round(((height_range[1]-height_range[0])*difficulty + height_range[0])/terrain.vertical_scale)


        min_size = round(size_range[0]/ terrain.horizontal_scale)
        max_size = round(size_range[1]/ terrain.horizontal_scale)

        discrete_start_x = start_x+platform_size
        discrete_start_y = start_y

        discrete_end_x = discrete_start_x +round(length_x/ terrain.horizontal_scale) - platform_size
        discrete_end_y = discrete_start_y +round(length_y/ terrain.horizontal_scale)

        num_rects = round((num_range[1]-num_range[0])*difficulty + num_range[0])

        for _ in range(num_rects):
            width = round(random.uniform(min_size, max_size))
            length = round(random.uniform(min_size, max_size))
            start_i = round(random.uniform(discrete_start_x, discrete_end_x-width))
            start_j = round(random.uniform(discrete_start_y, discrete_end_y-length))

            terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = random.uniform(-height//2, height)

        terrain.height_field_raw[start_x:start_x+platform_size , start_y:start_y+mid_y*2] = 0
        terrain.height_field_raw[discrete_end_x:discrete_end_x+platform_size , start_y:start_y+mid_y*2] = 0

        return terrain,goals,discrete_end_x+platform_size

    def stair(terrain,
                length_x=18.0,
                length_y=4.0,
                num_goals=8,
                start_x = 0,
                start_y = 0,
                platform_size=1.0, 
                difficulty = 0.5,
                #height_range=[0.1,0.2],
                height_range=[0.3,0.4],
                size_range=[0.5,0.6]
                ):

        goals = np.zeros((num_goals, 2))
        platform_size = round(platform_size/ terrain.horizontal_scale)
        per_x = (round(length_x/ terrain.horizontal_scale)- platform_size) // num_goals
        per_y = round(length_y/ terrain.horizontal_scale) // 2
        step_height = round(((height_range[1]-height_range[0])*difficulty + height_range[0])/terrain.vertical_scale)
        step_x = round(((size_range[0]-size_range[1])*difficulty +size_range[1])/terrain.horizontal_scale)
        total_step_height = 0

        dis_x = start_x + platform_size

        for i in range(num_goals):
            goals[i]=[dis_x+per_x*i,start_y+per_y]

        
        for i in range(num_goals):

            if(i < num_goals//2):
                total_step_height += step_height
            else:
                 total_step_height -= step_height
            # total_step_height += step_height
            terrain.height_field_raw[dis_x : dis_x + step_x, start_y : start_y + per_y*2] = total_step_height
            dis_x += step_x

        terrain.height_field_raw[start_x:start_x+platform_size,start_y:start_y + per_y*2] = 0
        terrain.height_field_raw[dis_x:start_x+per_x*num_goals+2*platform_size,start_y:start_y + per_y*2] = total_step_height

        return terrain,goals,start_x+per_x*num_goals

    def wave(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            amplitude_range=[0.05,0.1]
            ):   
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale) //2
        platform_size = round(1.5/ terrain.horizontal_scale)
        mid_x =  (round(length_x/ terrain.horizontal_scale) - platform_size)// num_goals

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+mid_x*i,start_y+mid_y]
        
        x_indices = np.arange(start_x, start_x + mid_x*num_goals + platform_size)
        amplitude = round(((amplitude_range[1]-amplitude_range[0])*difficulty + amplitude_range[0])/terrain.vertical_scale)
        wave_pattern = amplitude * np.sin(2 * np.pi * x_indices / length_x)

        for i, wave_height in enumerate(wave_pattern):
            terrain.height_field_raw[x_indices[i], start_y:start_y +mid_y*2] = wave_height

        terrain.height_field_raw[start_x :start_x + platform_size, start_y:start_y+ mid_y*2] = 0

        return terrain,goals,start_x+mid_x*num_goals

    def slope(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=3.0, 
            difficulty = 0.5,
            angle_range = [5.0,15.0],
            uphill=True
            ):    

        goals = np.zeros((num_goals, 2))
        length_x_grid = round((length_x - platform_size) / terrain.horizontal_scale)
        length_y_grid = round(length_y / terrain.horizontal_scale)
        platform_size = round(platform_size/ terrain.horizontal_scale)

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+length_x_grid/num_goals*i,start_y+length_y_grid//2]

        slope_angle = (angle_range[1]-angle_range[0])*difficulty + angle_range[0]
        angle_rad = math.radians(slope_angle)
        total_height = length_x * math.tan(angle_rad)
        total_height_units = total_height / terrain.vertical_scale

        # 确保平台区域的高度为0（平坦）
        terrain.height_field_raw[start_x:start_x + platform_size, start_y:start_y + length_y_grid] = 0

        # 创建斜坡
        for x in range(start_x + platform_size, start_x + platform_size + length_x_grid):
            progress = (x - (start_x + platform_size)) / length_x_grid
            if uphill:
                height = progress * total_height_units
            else:
                height = (1 - progress) * total_height_units
            terrain.height_field_raw[x, start_y:start_y + length_y_grid] = round(height)
        
        return terrain,goals,start_x + platform_size + length_x_grid

    def gap(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0,
            difficulty = 0.5,
            gap_height = 2.,
            gap_low_range = [0.3,0.4],
            ):
        
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale) //2
        mid_x =  round((length_x - platform_size)/ terrain.horizontal_scale) // num_goals
        platform_size = round(platform_size/ terrain.horizontal_scale)

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+mid_x*i,start_y+mid_y]

        gap_size = round(( (gap_low_range[0]-gap_low_range[1])*difficulty + gap_low_range[1] )/terrain.horizontal_scale)
        gap_dis_x = start_x + platform_size + gap_size
        gap_dis_y = start_y + mid_y
        
        for i in range(num_goals):
            terrain.height_field_raw[gap_dis_x :gap_dis_x + gap_size, gap_dis_y - mid_y:gap_dis_y + mid_y] = -round(gap_height / terrain.vertical_scale)
            gap_dis_x += 2*gap_size
        
        terrain.height_field_raw[start_x :start_x + platform_size, start_y :start_y + mid_y*2] = 0

        return terrain, goals,start_x+mid_x*num_goals
    
