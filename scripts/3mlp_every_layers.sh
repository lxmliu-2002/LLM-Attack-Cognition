#!/bin/bash

# nohup ./3mlp_every_layers.sh > ../src/tmp/nohup_20250731_151110_3.out 2>&1 &

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 开始批量训练 MLP 分类器${NC}"

# ========== 配置区 ==========
WORK_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/src"
CONDA_PATH="/home/liuxiuming/miniconda3"
ENV_NAME="t2i_fuzzer"
PYTHON_SCRIPT="mlp_every_layers.py"

HS_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/result/hiddenstates/ori"
MLP_OUTPUT_ROOT="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/result/mlp/all_layers"

TRAIN_JSONL="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/result/generate/train/train.jsonl"
EVAL_JSONL="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/result/generate/train/eval.jsonl"

LOG_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/src/log/3"
mkdir -p "$LOG_DIR" || { echo -e "${RED}错误：无法创建日志目录${NC}"; exit 1; }
mkdir -p "$MLP_OUTPUT_ROOT" || { echo -e "${RED}错误：无法创建 MLP 输出根目录${NC}"; exit 1; }

if [ ! -f "$TRAIN_JSONL" ]; then
    echo -e "${RED}错误：未找到训练集 JSONL 文件: $TRAIN_JSONL${NC}"
    exit 1
fi

if [ ! -f "$EVAL_JSONL" ]; then
    echo -e "${RED}错误：未找到验证集 JSONL 文件: $EVAL_JSONL${NC}"
    exit 1
fi
# =============================

cd "$WORK_DIR" || { echo -e "${RED}错误：无法进入目录 $WORK_DIR${NC}"; exit 1; }

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}错误：未找到 Python 脚本 $PYTHON_SCRIPT${NC}"
    exit 1
fi

source "$CONDA_PATH/etc/profile.d/conda.sh" || { echo -e "${RED}错误：无法加载 Conda${NC}"; exit 1; }
conda activate "$ENV_NAME" || { echo -e "${RED}错误：无法激活环境 $ENV_NAME${NC}"; exit 1; }
echo -e "${GREEN}✅ Conda 环境激活成功${NC}\n"

for train_npy in "$HS_DIR"/*_train.npy; do
    if [ ! -f "$train_npy" ]; then
        echo -e "${YELLOW}⚠️ 未找到任何 _train.npy 文件${NC}"
        continue
    fi

    base_name=$(basename "$train_npy" .npy)
    model_name="${base_name%_train}"
    eval_npy="$HS_DIR/${model_name}_eval.npy"

    if [ ! -f "$eval_npy" ]; then
        echo -e "${RED}❌ 缺少对应的 eval 文件: $eval_npy${NC}"
        continue
    fi

    MODEL_OUTPUT_DIR="$MLP_OUTPUT_ROOT/${model_name}"
    mkdir -p "$MODEL_OUTPUT_DIR" || { echo -e "${RED}错误：无法创建模型输出目录 $MODEL_OUTPUT_DIR${NC}"; continue; }

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/3_${TIMESTAMP}_${model_name}_mlp.log"

    echo -e "${YELLOW}🔍 处理模型: $model_name${NC}"
    echo -e "   📂 train: $(basename "$train_npy")"
    echo -e "   📂 eval:  $(basename "$eval_npy")"
    echo -e "   💾 模型保存路径: $MODEL_OUTPUT_DIR"
    echo -e "   📝 日志:  $LOG_FILE"

    python "$PYTHON_SCRIPT" \
        --train_hs "$train_npy" \
        --eval_hs "$eval_npy" \
        --train_jsonl "$TRAIN_JSONL" \
        --eval_jsonl "$EVAL_JSONL" \
        --output_dir "$MODEL_OUTPUT_DIR" > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}   ✅ 成功${NC}\n"
    else
        echo -e "${RED}   ❌ 失败，查看日志: $LOG_FILE${NC}\n"
    fi
    
    sleep 2
done

echo -e "${GREEN}🎉 所有模型的 MLP 训练任务已完成！模型保存在: $MLP_OUTPUT_ROOT${NC}"