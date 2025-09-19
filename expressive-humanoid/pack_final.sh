#!/bin/bash

# 最终打包脚本：包含所有代码，只精简motion和ckpt文件

set -e

PROJECT_ROOT="/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid"
PACKAGE_NAME="expressive-humanoid-final"
PACKAGE_DIR="${PROJECT_ROOT}/${PACKAGE_NAME}"
TARBALL="${PACKAGE_NAME}.tar.gz"

echo "开始最终打包..."

# 清理旧目录
if [ -d "$PACKAGE_DIR" ]; then
    rm -rf "$PACKAGE_DIR"
fi

mkdir -p "$PACKAGE_DIR"

echo "复制所有代码文件..."
# 复制整个项目，排除logs目录
rsync -av --exclude='logs/' --exclude='__pycache__/' --exclude='*.pyc' \
    --exclude='wandb/' --exclude='.git/' \
    "$PROJECT_ROOT/" "$PACKAGE_DIR/"

echo "创建空的logs目录结构..."
mkdir -p "$PACKAGE_DIR/legged_gym/logs/h1"

echo "清理ASE目录，只保留EasyWalking需要的motion文件..."

# 清理retarget_npy中不需要的motion文件
MOTION_IDS=("02_01" "02_02" "05_01" "06_01" "07_01" "07_02" "07_03")

echo "清理retarget_npy目录..."
find "$PACKAGE_DIR/ASE/ase/poselib/data/retarget_npy/" -name "*.npy" \
    ! -name "02_01.npy" ! -name "02_02.npy" ! -name "05_01.npy" ! -name "06_01.npy" ! -name "07_01.npy" ! -name "07_02.npy" ! -name "07_03.npy" \
    ! -name "02_01_key_bodies.npy" ! -name "02_02_key_bodies.npy" ! -name "05_01_key_bodies.npy" ! -name "06_01_key_bodies.npy" ! -name "07_01_key_bodies.npy" ! -name "07_02_key_bodies.npy" ! -name "07_03_key_bodies.npy" \
    -delete 2>/dev/null || true

echo "清理pkl目录，只保留EasyWalking的pkl文件..."
find "$PACKAGE_DIR/ASE/ase/poselib/data/pkl/" -name "*.pkl" ! -name "motions_easywalk.pkl" -delete 2>/dev/null || true

echo "清理其他数据目录..."
# 删除不需要的数据目录
rm -rf "$PACKAGE_DIR/ASE/ase/poselib/data/npy" 2>/dev/null || true
rm -rf "$PACKAGE_DIR/ASE/ase/poselib/data/cmu_fbx_all" 2>/dev/null || true
rm -rf "$PACKAGE_DIR/ASE/ase/poselib/data/amass_test" 2>/dev/null || true
rm -rf "$PACKAGE_DIR/ASE/ase/poselib/data/tpose" 2>/dev/null || true

echo "✅ ASE目录清理完成，只保留EasyWalking相关文件"

echo "计算文件大小..."
TOTAL_SIZE=$(du -sh "$PACKAGE_DIR" | cut -f1)
echo "打包目录大小: $TOTAL_SIZE"

echo "创建压缩包..."
cd "$PROJECT_ROOT"
tar -czf "$TARBALL" "$PACKAGE_NAME/"

FINAL_SIZE=$(du -sh "$TARBALL" | cut -f1)
echo "压缩包大小: $FINAL_SIZE"

echo ""
echo "✅ 最终打包完成！"
echo "📦 压缩包: $PROJECT_ROOT/$TARBALL"
echo "📁 解压目录: $PACKAGE_DIR"
echo ""
echo "📋 包含内容:"
echo "   - 所有代码文件（legged_gym, rsl_rl, ASE, challenging_terrain, isaacgym）"
echo "   - 只保留EasyWalking的7个motion文件（14个npy文件）"
echo "   - 只保留EasyWalking的pkl文件"
echo "   - 空的logs目录结构"
echo ""
echo "❌ 排除内容:"
echo "   - 所有ckpt模型文件"
echo "   - 其他所有motion数据文件"
echo "   - wandb日志"
echo "   - git历史"
echo "   - 缓存文件"
echo ""
echo "🚀 使用方法:"
echo "   1. tar -xzf $TARBALL"
echo "   2. cd $PACKAGE_NAME"
echo "   3. 直接运行训练或推理"