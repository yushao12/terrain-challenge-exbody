#!/bin/bash

# G1 AMP训练示例脚本
# 使用方式：./train_g1_amp_example.sh

# 基本参数
TASK="g1_mimic_amp"                    # 任务名称
PROJ_NAME="g1_mimic_amp"               # 项目名称
EXPT_ID="g1_amp_$(date +%Y%m%d_%H%M%S)"  # 实验ID（自动生成时间戳）
MOTION_NAME="motions_g1_all.yaml"      # G1的motion配置文件
MOTION_TYPE="yaml"                     # motion类型

# 训练参数
MAX_ITERATIONS=1500                    # 最大训练迭代数
NUM_ENVS=4096                          # 环境数量
SEED=1                                 # 随机种子

# 运行训练
cd /home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym/legged_gym/scripts

python train.py \
    --task $TASK \
    --proj_name $PROJ_NAME \
    --exptid $EXPT_ID \
    --motion_name $MOTION_NAME \
    --motion_type $MOTION_TYPE \
    --max_iterations $MAX_ITERATIONS \
    --num_envs $NUM_ENVS \
    --seed $SEED \
    --headless \
    --no_wandb

echo "G1 AMP训练完成！"
echo "日志保存在: logs/$PROJ_NAME/$EXPT_ID/"
echo "模型保存在: logs/$PROJ_NAME/$EXPT_ID/model_*.pt"