#!/bin/bash

# nohup ./scripts/1extra_hs.sh > ../src/tmp/nohup_20250730_184515_1.out 2>&1 &


export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 开始提取隐藏状态任务${NC}"

# ========== 配置区 ==========
WORK_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/src"
CONDA_PATH="/home/liuxiuming/miniconda3"
ENV_NAME="t2i_fuzzer"
PYTHON_SCRIPT="extra_hs.py"

INPUT_TRAIN="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/data/mlp/train/train.jsonl"
INPUT_EVAL="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/data/mlp/train/eval.jsonl"

OUTPUT_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/result/hiddenstates/ori"
mkdir -p "$OUTPUT_DIR" || {
    echo -e "${RED}错误：无法创建输出目录 $OUTPUT_DIR${NC}"
    exit 1
}

LOG_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/src/log"
mkdir -p "$LOG_DIR" || {
    echo -e "${RED}错误：无法创建日志目录 $LOG_DIR${NC}"
    exit 1
}

MODELS=(
    "/mnt/sdb/models/Qwen/Qwen3-8B"
    "/mnt/sdb/models/Qwen/Qwen3-4B"
    "/mnt/sdb/models/Qwen/Qwen2.5-7B-Instruct"
    "/mnt/sdb/models/Qwen/Qwen2.5-1.5B-Instruct"
    "/mnt/sdb/models/llama/Llama-3.1-8B-Instruct"
    "/mnt/sdb/models/llama/Llama-3.2-3B-Instruct"
)
# =============================

echo -e "${YELLOW}📁 切换工作目录: $WORK_DIR${NC}"
cd "$WORK_DIR" || {
    echo -e "${RED}错误：无法进入目录 $WORK_DIR${NC}"
    exit 1
}

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}错误：未找到 Python 脚本 $PYTHON_SCRIPT${NC}"
    exit 1
fi

for file in "$INPUT_TRAIN" "$INPUT_EVAL"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}错误：未找到输入文件 $file${NC}"
        exit 1
    fi
done

source "$CONDA_PATH/etc/profile.d/conda.sh" || {
    echo -e "${RED}错误：无法加载 Conda 配置文件${NC}"
    exit 1
}

echo -e "${YELLOW}🔄 激活 Conda 环境: $ENV_NAME${NC}"
conda activate "$ENV_NAME" || {
    echo -e "${RED}错误：无法激活 Conda 环境 '$ENV_NAME'${NC}"
    exit 1
}

echo -e "${GREEN}✅ Conda 环境激活成功，当前 Python: $(basename $(which python))${NC}\n"

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo -e "${YELLOW}🔧 正在处理模型: $MODEL_NAME${NC}"

    declare -a DATASETS=("$INPUT_TRAIN" "$INPUT_EVAL")
    DATASET_NAMES=("train" "eval")

    for i in "${!DATASETS[@]}"; do
        INPUT_FILE="${DATASETS[i]}"
        DATASET_NAME="${DATASET_NAMES[i]}"

        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        LOG_FILE="$LOG_DIR/${TIMESTAMP}_${MODEL_NAME}_${DATASET_NAME}.log"

        echo -e "${GREEN}  ➤ 提取 ${DATASET_NAME} 数据...${NC}"
        echo -e "日志将保存到: $LOG_FILE"

        python "$PYTHON_SCRIPT" \
            --input "$INPUT_FILE" \
            --model_path "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" >> "$LOG_FILE" 2>&1

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}    ✅ ${DATASET_NAME} 提取完成${NC}"
        else
            echo -e "${RED}    ❌ ${DATASET_NAME} 提取失败: $MODEL_NAME${NC}"
        fi
    done

    echo -e "${YELLOW}✅ 模型 $MODEL_NAME 的处理完成${NC}\n"
done

echo -e "${GREEN}🎉 所有模型的隐藏状态提取任务已完成！${NC}"