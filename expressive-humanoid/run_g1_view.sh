#!/bin/bash

# 简单的G1机器人查看脚本

echo "启动G1机器人可视化..."

# 设置环境变量
export PYTHONPATH="/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym:$PYTHONPATH"

# 进入项目目录
cd /home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid

# 直接运行训练脚本进行可视化（不训练，只查看）
echo "运行G1机器人可视化..."
python legged_gym/legged_gym/scripts/train.py --task=g1 --num_envs=1 --max_iterations=100 --no_wandb g1_view

echo "完成!"