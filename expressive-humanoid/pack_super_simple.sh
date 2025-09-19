#!/bin/bash

# 超级简单打包脚本：排除所有不需要的数据，只保留EasyWalking

set -e

PROJECT_ROOT="/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid"
PACKAGE_NAME="expressive-humanoid-clean"
PACKAGE_DIR="${PROJECT_ROOT}/${PACKAGE_NAME}"
TARBALL="${PACKAGE_NAME}.tar.gz"

echo "开始超级简单打包..."

# 清理旧目录
if [ -d "$PACKAGE_DIR" ]; then
    rm -rf "$PACKAGE_DIR"
fi

mkdir -p "$PACKAGE_DIR"

echo "复制所有代码文件（排除大数据目录）..."

# 复制项目，排除所有大数据目录
rsync -av \
    --exclude='logs/' \
    --exclude='wandb/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='ASE/ase/poselib/data/npy/' \
    --exclude='ASE/ase/poselib/data/retarget_npy/' \
    --exclude='ASE/ase/poselib/data/cmu_fbx_all/' \
    --exclude='ASE/ase/poselib/data/amass_test/' \
    --exclude='ASE/ase/poselib/data/pkl/' \
    --exclude='ASE/ase/poselib/data/tpose/' \
    --exclude='*.tar.gz' \
    "$PROJECT_ROOT/" "$PACKAGE_DIR/"

echo "创建logs目录结构..."
mkdir -p "$PACKAGE_DIR/legged_gym/logs/h1"

echo "复制EasyWalking数据..."

# 创建数据目录
mkdir -p "$PACKAGE_DIR/ASE/ase/poselib/data/retarget_npy"
mkdir -p "$PACKAGE_DIR/ASE/ase/poselib/data/pkl"

# 复制EasyWalking的7个motion文件
MOTION_IDS=("02_01" "02_02" "05_01" "06_01" "07_01" "07_02" "07_03")

for motion_id in "${MOTION_IDS[@]}"; do
    # 复制主motion文件
    if [ -f "$PROJECT_ROOT/ASE/ase/poselib/data/retarget_npy/${motion_id}.npy" ]; then
        cp "$PROJECT_ROOT/ASE/ase/poselib/data/retarget_npy/${motion_id}.npy" \
           "$PACKAGE_DIR/ASE/ase/poselib/data/retarget_npy/"
        echo "  复制 ${motion_id}.npy"
    fi
    
    # 复制关键身体位置文件
    if [ -f "$PROJECT_ROOT/ASE/ase/poselib/data/retarget_npy/${motion_id}_key_bodies.npy" ]; then
        cp "$PROJECT_ROOT/ASE/ase/poselib/data/retarget_npy/${motion_id}_key_bodies.npy" \
           "$PACKAGE_DIR/ASE/ase/poselib/data/retarget_npy/"
        echo "  复制 ${motion_id}_key_bodies.npy"
    fi
done

# 复制pkl文件
if [ -f "$PROJECT_ROOT/ASE/ase/poselib/data/pkl/motions_easywalk.pkl" ]; then
    cp "$PROJECT_ROOT/ASE/ase/poselib/data/pkl/motions_easywalk.pkl" \
       "$PACKAGE_DIR/ASE/ase/poselib/data/pkl/"
    echo "  复制 motions_easywalk.pkl"
fi

echo "计算文件大小..."
TOTAL_SIZE=$(du -sh "$PACKAGE_DIR" | cut -f1)
echo "打包目录大小: $TOTAL_SIZE"

echo "创建压缩包..."
cd "$PROJECT_ROOT"
tar -czf "$TARBALL" "$PACKAGE_NAME/"

FINAL_SIZE=$(du -sh "$TARBALL" | cut -f1)
echo "压缩包大小: $FINAL_SIZE"

echo ""
echo "✅ 超级简单打包完成！"
echo "📦 压缩包: $PROJECT_ROOT/$TARBALL"
echo "📁 解压目录: $PACKAGE_DIR"
echo ""
echo "📋 包含内容:"
echo "   - 所有代码文件"
echo "   - 只保留EasyWalking的7个motion文件（14个npy文件）"
echo "   - 只保留EasyWalking的pkl文件"
echo "   - 空的logs/h1目录"
echo ""
echo "❌ 排除内容:"
echo "   - 所有其他motion数据文件"
echo "   - 所有ckpt文件"
echo "   - 大数据目录"
echo ""
echo "🚀 使用方法:"
echo "   1. tar -xzf $TARBALL"
echo "   2. cd $PACKAGE_NAME"
echo "   3. 直接运行训练或推理"