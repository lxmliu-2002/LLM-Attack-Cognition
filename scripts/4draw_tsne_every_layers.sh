#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🎨 开始批量生成 t-SNE 可视化图像${NC}"

# ========== 配置区 ==========
WORK_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/src"
CONDA_PATH="/home/liuxiuming/miniconda3"
ENV_NAME="t2i_fuzzer"
PYTHON_SCRIPT="draw_tsne_every_layers.py"

HS_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/result/hiddenstates/ori"
OUTPUT_IMG_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/result/imgs/hiddenstates/train"
JSONL_FILE="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/result/generate/train/train.jsonl"
LOG_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/src/log/4"

mkdir -p "$OUTPUT_IMG_DIR" || { echo -e "${RED}❌ 无法创建图像输出目录${NC}"; exit 1; }
mkdir -p "$LOG_DIR" || { echo -e "${RED}❌ 无法创建日志目录${NC}"; exit 1; }


if [ ! -f "$JSONL_FILE" ]; then
    echo -e "${RED}❌ 未找到 JSONL 文件: $JSONL_FILE${NC}"
    exit 1
fi

cd "$WORK_DIR" || { echo -e "${RED}❌ 无法进入工作目录: $WORK_DIR${NC}"; exit 1; }

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}❌ 未找到 Python 脚本: $PYTHON_SCRIPT${NC}"
    exit 1
fi

source "$CONDA_PATH/etc/profile.d/conda.sh" || { echo -e "${RED}❌ 无法加载 Conda${NC}"; exit 1; }
conda activate "$ENV_NAME" || { echo -e "${RED}❌ 无法激活 Conda 环境: $ENV_NAME${NC}"; exit 1; }
echo -e "${GREEN}✅ Conda 环境激活成功: $ENV_NAME${NC}"
# =============================

echo -e "${YELLOW}🔍 开始处理所有 _train.npy 文件...${NC}"

for train_npy in "$HS_DIR"/*_train.npy; do
    if [ ! -f "$train_npy" ]; then
        echo -e "${YELLOW}⚠️ 未找到任何 _train.npy 文件${NC}"
        continue
    fi

    base_name=$(basename "$train_npy" .npy)
    model_name="${base_name%_train}"

    LOG_FILE="$LOG_DIR/4_${model_name}_tsne.log"
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    echo -e "\n${YELLOW}🖼️  处理模型: $model_name${NC}"
    echo -e "   📂 输入: $train_npy"
    echo -e "   📂 输出图像: $OUTPUT_IMG_DIR/$model_name.png"
    echo -e "   📝 日志: $LOG_FILE"

    # 运行 Python 脚本
    python "$PYTHON_SCRIPT" \
        --hs "$train_npy" \
        --jsonl "$JSONL_FILE" \
        --output_dir "$OUTPUT_IMG_DIR" > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}   ✅ 成功生成图像${NC}"
    else
        echo -e "${RED}   ❌ 失败，查看日志: $LOG_FILE${NC}"
    fi

    sleep 1
done

echo -e "\n${GREEN}🎉 所有 t-SNE 可视化任务已完成！${NC}"
echo -e "🖼️  图像保存在: $OUTPUT_IMG_DIR"
echo -e "📄 日志保存在: $LOG_DIR"