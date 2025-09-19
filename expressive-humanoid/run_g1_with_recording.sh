#!/bin/bash

# 运行G1机器人并自动录屏

echo "启动G1机器人可视化并录屏..."

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

# 创建视频文件名
timestamp=$(date +"%Y%m%d_%H%M%S")
video_file="g1_robot_demo_${timestamp}.mp4"

echo "开始录屏到文件: $video_file"

# 启动录屏（在后台）
ffmpeg -f x11grab -s 1024x768 -r 30 -i :99 -c:v libx264 -preset fast -crf 18 -y "$video_file" &
FFMPEG_PID=$!

# 等待录屏启动
sleep 2

# 运行G1机器人可视化
echo "运行G1机器人可视化..."
python legged_gym/scripts/train.py --task=g1 --headless=False --num_envs=1 --max_iterations=1

# 停止录屏
echo "停止录屏..."
kill $FFMPEG_PID
sleep 2

# 清理xvfb进程
echo "清理虚拟显示..."
kill $XVFB_PID

echo "✓ 完成! 视频保存为: $video_file"
echo "您可以使用以下命令查看视频:"
echo "  ffplay $video_file"
echo "  vlc $video_file"