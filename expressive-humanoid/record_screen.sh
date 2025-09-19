#!/bin/bash

# 录屏脚本

echo "开始录屏..."

# 创建视频文件名
timestamp=$(date +"%Y%m%d_%H%M%S")
video_file="g1_robot_demo_${timestamp}.mp4"

echo "录屏到文件: $video_file"
echo "按 Ctrl+C 停止录屏"

# 开始录屏
ffmpeg -f x11grab -s 1024x768 -r 30 -i :99 -c:v libx264 -preset fast -crf 18 -y "$video_file"

echo "✓ 录屏完成! 视频保存为: $video_file"
echo "您可以使用以下命令查看视频:"
echo "  ffplay $video_file"
echo "  vlc $video_file"