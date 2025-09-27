# H1_2机器人设置说明

## 概述
本文档描述了如何将H1_2机器人集成到expressive-humanoid项目中，参考G1机器人的处理方式。

## 文件结构

### 1. 机器人资源文件
```
legged_gym/resources/robots/h1_2/
├── h1_2_fix_arm.urdf          # H1_2的URDF文件
├── h1_2.png                   # 机器人图片
├── meshes/                    # 3D网格文件
└── scene.xml                  # 场景配置
```

### 2. 环境配置文件
```
legged_gym/envs/h1_2/
├── __init__.py                # 模块初始化
├── h1_2_mimic_config.py       # 基础配置
├── h1_2_mimic_amp_config.py   # AMP配置
├── h1_2_mimic.py              # 基础环境类
├── h1_2_mimic_amp.py          # AMP环境类
└── h1_2_mimic_view_motion.py  # Motion可视化环境
```

### 3. Motion数据处理
```
ASE/ase/poselib/
├── define_h1_2_key_bodies.py      # 定义关键关节
├── generate_h1_2_key_bodies.py    # 生成key body数据
└── retarget_h1_2/
    └── retarget_to_h1_2.py        # Motion重定向脚本

ASE/ase/data/
└── motions_h1_2_all.yaml          # H1_2 motion配置
```

### 4. 工具脚本
```
├── quick_h1_2_view.py             # 快速查看H1_2配置
└── legged_gym/scripts/play_h1_2.py # H1_2播放脚本
```

## 配置特点

### H1_2机器人特性
- **DOF数量**: 12个自由度（与G1相同）
- **关节结构**: 左右腿各6个关节（hip_yaw, hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll）
- **URDF文件**: `h1_2_fix_arm.urdf`
- **关键关节**: 7个（pelvis + 6个腿部关键关节）

### 观测空间配置
- **n_proprio**: 51维（与原项目terrain-challenge保持一致）
- **n_scan**: 132维
- **n_priv**: 9维（3+3+3）
- **n_priv_latent**: 29维（4+1+12+12）
- **总观测维度**: 731维

### 控制参数
- **控制类型**: P控制
- **动作缩放**: 0.25
- **降采样**: 4
- **刚度参数**: 针对不同关节类型设置不同值
- **阻尼参数**: 针对不同关节类型设置不同值

## 注册的任务

在`legged_gym/envs/__init__.py`中注册了以下H1_2任务：

1. **h1_2_mimic**: 基础mimic任务
2. **h1_2_mimic_amp**: AMP训练任务
3. **h1_2_view**: Motion可视化任务

## 使用方法

### 1. 快速查看配置
```bash
python quick_h1_2_view.py
```

### 2. 播放H1_2策略
```bash
python legged_gym/scripts/play_h1_2.py --task h1_2_mimic_amp --mode policy
```

### 3. 可视化motion数据
```bash
python legged_gym/scripts/play_h1_2.py --mode motion --motion motions_h1_2_all.yaml
```

### 4. 训练H1_2
```bash
python legged_gym/scripts/train.py h1_2_debug --task h1_2_mimic_amp
```

## 与G1的对比

| 特性 | G1 | H1_2 |
|------|----|----- |
| DOF数量 | 12 | 12 |
| URDF文件 | g1_12dof_with_hand.urdf | h1_2_fix_arm.urdf |
| 关键关节数 | 7 | 7 |
| 观测维度 | 731 | 731 |
| 控制参数 | 相同 | 相同 |

## 注意事项

1. **Motion数据**: 需要为H1_2生成相应的motion数据和key body文件
2. **URDF兼容性**: H1_2的URDF结构与G1相似，但需要验证关节顺序
3. **训练策略**: 可以使用原项目的H1 checkpoint进行迁移学习
4. **环境配置**: 所有配置都基于G1的配置进行调整

## 下一步工作

1. 生成H1_2的motion数据
2. 测试H1_2环境的正确性
3. 验证与原项目H1 checkpoint的兼容性
4. 进行H1_2的训练和测试