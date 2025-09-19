#!/bin/bash

# 使用xvfb运行G1机器人可视化

echo "使用虚拟显示启动G1机器人可视化..."

# 设置虚拟显示
export DISPLAY=:99

# 启动xvfb
echo "启动虚拟显示..."
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# 等待xvfb启动
sleep 3

# 检查xvfb是否启动成功
if ! ps -p $XVFB_PID > /dev/null; then
    echo "错误: xvfb启动失败"
    exit 1
fi

echo "✓ 虚拟显示启动成功 (PID: $XVFB_PID)"

# 设置环境变量
export PYTHONPATH="/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid/legged_gym:$PYTHONPATH"

# 进入项目目录
cd /home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid

# 运行G1机器人可视化
echo "运行G1机器人可视化..."
python legged_gym/scripts/train.py --task=g1 --headless=False --num_envs=1 --max_iterations=1

# 清理xvfb进程
echo "清理虚拟显示..."
kill $XVFB_PID

echo "✓ 完成!"