#!/bin/bash

# 健壮的G1机器人录屏脚本

echo "启动G1机器人可视化并录屏..."

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

# 创建视频文件名
timestamp=$(date +"%Y%m%d_%H%M%S")
video_file="g1_robot_demo_${timestamp}.mp4"

echo "开始录屏到文件: $video_file"

# 启动录屏（在后台）
ffmpeg -f x11grab -s 1024x768 -r 30 -i :99 -c:v libx264 -preset fast -crf 18 -y "$video_file" &
FFMPEG_PID=$!

# 等待录屏启动
sleep 2

# 设置清理函数
cleanup() {
    echo "清理资源..."
    kill $FFMPEG_PID 2>/dev/null
    kill $XVFB_PID 2>/dev/null
    echo "✓ 录屏完成! 视频保存为: $video_file"
    exit 0
}

# 捕获中断信号
trap cleanup SIGINT SIGTERM

# 运行G1机器人可视化
echo "运行G1机器人可视化..."
echo "按 Ctrl+C 停止演示和录屏"

python legged_gym/scripts/train.py --task=g1 --headless=False --num_envs=1 --max_iterations=1

# 正常结束时的清理
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