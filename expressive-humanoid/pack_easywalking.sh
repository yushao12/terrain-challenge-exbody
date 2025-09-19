#!/bin/bash

# Expressive Humanoid - EasyWalking 最小化打包脚本
# 排除ckpt文件和不必要的文件，只打包easywalking.yaml相关的核心文件

set -e

# 配置
PROJECT_ROOT="/home/ouyangyu/hdd2_workspace/exbody/expressive-humanoid"
PACKAGE_NAME="expressive-humanoid-easywalking"
PACKAGE_DIR="${PROJECT_ROOT}/${PACKAGE_NAME}"
TARBALL="${PACKAGE_NAME}.tar.gz"

echo "开始打包 Expressive Humanoid EasyWalking 项目..."

# 清理旧的打包目录
if [ -d "$PACKAGE_DIR" ]; then
    echo "清理旧的打包目录..."
    rm -rf "$PACKAGE_DIR"
fi

# 创建打包目录
mkdir -p "$PACKAGE_DIR"

echo "复制核心代码文件..."

# 1. 复制legged_gym核心代码（排除logs和wandb）
mkdir -p "$PACKAGE_DIR/legged_gym"
rsync -av --exclude='logs/' --exclude='wandb/' --exclude='__pycache__/' --exclude='*.pyc' \
    "$PROJECT_ROOT/legged_gym/" "$PACKAGE_DIR/legged_gym/"

# 2. 复制rsl_rl核心代码
mkdir -p "$PACKAGE_DIR/rsl_rl"
rsync -av --exclude='__pycache__/' --exclude='*.pyc' \
    "$PROJECT_ROOT/rsl_rl/" "$PACKAGE_DIR/rsl_rl/"

# 3. 复制ASE核心代码（排除数据文件）
mkdir -p "$PACKAGE_DIR/ASE"
rsync -av --exclude='__pycache__/' --exclude='*.pyc' \
    --exclude='ase/poselib/data/npy/' \
    --exclude='ase/poselib/data/retarget_npy/' \
    --exclude='ase/poselib/data/cmu_fbx_all/' \
    --exclude='ase/poselib/data/amass_test/' \
    --exclude='ase/poselib/data/pkl/' \
    --exclude='ase/poselib/data/tpose/' \
    "$PROJECT_ROOT/ASE/" "$PACKAGE_DIR/ASE/"

# 3.5. 复制challenging_terrain地形数据
if [ -d "$PROJECT_ROOT/challenging_terrain" ]; then
    echo "复制Challenging Terrain..."
    mkdir -p "$PACKAGE_DIR/challenging_terrain"
    rsync -av --exclude='__pycache__/' --exclude='*.pyc' \
        "$PROJECT_ROOT/challenging_terrain/" "$PACKAGE_DIR/challenging_terrain/"
fi

# 4. 复制isaacgym（如果需要的话）
if [ -d "$PROJECT_ROOT/isaacgym" ]; then
    echo "复制IsaacGym..."
    mkdir -p "$PACKAGE_DIR/isaacgym"
    rsync -av --exclude='__pycache__/' --exclude='*.pyc' \
        "$PROJECT_ROOT/isaacgym/" "$PACKAGE_DIR/isaacgym/"
fi

echo "复制EasyWalking相关数据文件..."

# 5. 复制EasyWalking配置文件
mkdir -p "$PACKAGE_DIR/ASE/ase/poselib/data/configs"
cp "$PROJECT_ROOT/ASE/ase/poselib/data/configs/motions_easywalk.yaml" \
   "$PACKAGE_DIR/ASE/ase/poselib/data/configs/"

# 6. 复制EasyWalking相关的motion数据文件
mkdir -p "$PACKAGE_DIR/ASE/ase/poselib/data/retarget_npy"
mkdir -p "$PACKAGE_DIR/ASE/ase/poselib/data/pkl"
# 从motions_easywalk.yaml中提取的motion IDs
MOTION_IDS=("02_01" "02_02" "05_01" "06_01" "07_01" "07_02" "07_03")

for motion_id in "${MOTION_IDS[@]}"; do
    # 复制主motion文件
    if [ -f "$PROJECT_ROOT/ASE/ase/poselib/data/retarget_npy/${motion_id}.npy" ]; then
        cp "$PROJECT_ROOT/ASE/ase/poselib/data/retarget_npy/${motion_id}.npy" \
           "$PACKAGE_DIR/ASE/ase/poselib/data/retarget_npy/"
        echo "  复制 ${motion_id}.npy"
    else
        echo "  警告: ${motion_id}.npy 不存在"
    fi
    
    # 复制关键身体位置文件
    if [ -f "$PROJECT_ROOT/ASE/ase/poselib/data/retarget_npy/${motion_id}_key_bodies.npy" ]; then
        cp "$PROJECT_ROOT/ASE/ase/poselib/data/retarget_npy/${motion_id}_key_bodies.npy" \
           "$PACKAGE_DIR/ASE/ase/poselib/data/retarget_npy/"
        echo "  复制 ${motion_id}_key_bodies.npy"
    else
        echo "  警告: ${motion_id}_key_bodies.npy 不存在"
    fi
done

# 复制EasyWalking的pkl文件
if [ -f "$PROJECT_ROOT/ASE/ase/poselib/data/pkl/motions_easywalk.pkl" ]; then
    cp "$PROJECT_ROOT/ASE/ase/poselib/data/pkl/motions_easywalk.pkl" \
       "$PACKAGE_DIR/ASE/ase/poselib/data/pkl/"
    echo "  复制 motions_easywalk.pkl"
else
    echo "  警告: motions_easywalk.pkl 不存在"
fi

# 删除可能被rsync复制的其他npy文件
echo "清理不需要的motion文件..."
find "$PACKAGE_DIR/ASE/ase/poselib/data/retarget_npy/" -name "*.npy" \
    ! -name "02_01.npy" ! -name "02_02.npy" ! -name "05_01.npy" ! -name "06_01.npy" ! -name "07_01.npy" ! -name "07_02.npy" ! -name "07_03.npy" \
    ! -name "02_01_key_bodies.npy" ! -name "02_02_key_bodies.npy" ! -name "05_01_key_bodies.npy" ! -name "06_01_key_bodies.npy" ! -name "07_01_key_bodies.npy" ! -name "07_02_key_bodies.npy" ! -name "07_03_key_bodies.npy" \
    -delete 2>/dev/null || true

# 7. 复制必要的配置文件
if [ -f "$PROJECT_ROOT/ASE/ase/poselib/data/configs/retarget_to_h1.json" ]; then
    cp "$PROJECT_ROOT/ASE/ase/poselib/data/configs/retarget_to_h1.json" \
       "$PACKAGE_DIR/ASE/ase/poselib/data/configs/"
fi

# 8. 复制项目根目录的重要文件
cp "$PROJECT_ROOT/README.md" "$PACKAGE_DIR/" 2>/dev/null || true
cp "$PROJECT_ROOT/LICENSE" "$PACKAGE_DIR/" 2>/dev/null || true
cp "$PROJECT_ROOT/requirements.txt" "$PACKAGE_DIR/" 2>/dev/null || true

echo "创建部署说明文件..."

# 9. 创建部署说明
cat > "$PACKAGE_DIR/DEPLOYMENT.md" << 'EOF'
# Expressive Humanoid EasyWalking 部署说明

## 文件结构
- `legged_gym/`: 主要训练和推理代码
- `rsl_rl/`: 强化学习算法
- `ASE/`: 动作重定向和数据处理
- `challenging_terrain/`: 地形挑战数据
- `isaacgym/`: 物理仿真环境（如果包含）

## EasyWalking Motion 数据
包含以下7个walking motion：
- 02_01, 02_02, 05_01, 06_01, 07_01, 07_02, 07_03

## 使用方法
1. 安装依赖：`pip install -r requirements.txt`
2. 安装IsaacGym（如果需要）
3. 训练模型：
   ```bash
   cd legged_gym/legged_gym/scripts
   python train.py xxx-xx-easywalking --motion_name motions_easywalk.yaml --device cuda:0
   ```
4. 运行推理：
   ```bash
   cd legged_gym/legged_gym/scripts
   python play.py xxx-xx --motion_name motions_easywalk.yaml
   ```
5. 保存部署模型：
   ```bash
   python save_jit.py --exptid xxx-xx
   ```

## 注意
- 此包不包含训练好的模型文件（ckpt）
- 需要单独提供训练好的模型
- 确保IsaacGym正确安装
EOF

echo "创建requirements.txt（如果不存在）..."
if [ ! -f "$PACKAGE_DIR/requirements.txt" ]; then
    cat > "$PACKAGE_DIR/requirements.txt" << 'EOF'
torch>=1.10.0
torchvision>=0.11.0
torchaudio>=0.10.0
numpy<1.24
pydelatin
wandb
tqdm
opencv-python
ipdb
pyfqmr
flask
dill
gdown
EOF
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
echo "✅ 打包完成！"
echo "📦 压缩包: $PROJECT_ROOT/$TARBALL"
echo "📁 解压目录: $PACKAGE_DIR"
echo ""
echo "📋 包含内容:"
echo "   - legged_gym核心代码（不含logs）"
echo "   - rsl_rl强化学习算法"
echo "   - ASE动作重定向代码"
echo "   - challenging_terrain地形数据"
echo "   - EasyWalking配置文件 (motions_easywalk.yaml)"
echo "   - 7个walking motion数据文件 (14个npy文件)"
echo "   - EasyWalking pkl文件 (motions_easywalk.pkl)"
echo "   - 部署说明文档"
echo ""
echo "❌ 排除内容:"
echo "   - 所有ckpt模型文件"
echo "   - 训练日志和wandb数据"
echo "   - 其他motion数据文件"
echo "   - 缓存和临时文件"
echo ""
echo "🚀 使用方法:"
echo "   1. tar -xzf $TARBALL"
echo "   2. cd $PACKAGE_NAME"
echo "   3. 安装依赖: pip install -r requirements.txt"
echo "   4. 提供训练好的模型文件"
echo "   5. 运行推理脚本"