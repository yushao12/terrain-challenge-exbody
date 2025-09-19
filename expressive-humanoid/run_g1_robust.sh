#!/bin/bash

# 健壮的G1机器人运行脚本

echo "启动G1机器人可视化..."

# 设置虚拟显示
export DISPLAY=:99

# 清理可能存在的锁文件
echo "清理显示锁文件..."
rm -f /tmp/.X99-lock

# 检查是否已有xvfb在运行
if pgrep -f "Xvfb :99" > /dev/null; then
    echo "发现已有Xvfb在运行，停止它..."
    pkill -f "Xvfb :99"
    sleep 2
fi

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
echo "按 Ctrl+C 停止演示"

# 设置清理函数
cleanup() {
    echo "清理资源..."
    kill $XVFB_PID 2>/dev/null
    exit 0
}

# 捕获中断信号
trap cleanup SIGINT SIGTERM

# 运行演示
python legged_gym/scripts/train.py --task=g1 --headless=False --num_envs=1 --max_iterations=1

# 清理xvfb进程
echo "清理虚拟显示..."
kill $XVFB_PID

echo "✓ 演示完成!"