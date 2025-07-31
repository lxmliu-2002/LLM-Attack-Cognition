#!/bin/bash

# nohup ./scripts/2generate.sh > ../src/tmp/nohup_20250730_201134_2.out 2>&1 &

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 开始安全原地生成任务${NC}"

# ========== 配置区 ==========
WORK_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/src"
CONDA_PATH="/home/liuxiuming/miniconda3"
ENV_NAME="t2i_fuzzer"
PYTHON_SCRIPT="generate.py"

DATA_ROOT="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/data/mlp"

TMP_WORK_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/src/tmp"

TMP_WORK_DIR=$(mktemp -d -p "$TMP_WORK_DIR" mlp_tmp_XXXXXX)
echo -e "${YELLOW}📁 临时工作目录: $TMP_WORK_DIR${NC}"

RESULT_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/result/generate"
mkdir -p "$RESULT_DIR" || { echo -e "${RED}错误：无法创建输出目录${NC}"; exit 1; }

LOG_DIR="/mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/src/log"
mkdir -p "$LOG_DIR" || { echo -e "${RED}错误：无法创建日志目录${NC}"; exit 1; }

MODELS=(
    "/mnt/sdb/models/Qwen/Qwen3-4B"
    "/mnt/sdb/models/Qwen/Qwen3-8B"
    "/mnt/sdb/models/Qwen/Qwen2.5-7B-Instruct"
    "/mnt/sdb/models/llama/Llama-3.1-8B-Instruct"
    "/mnt/sdb/models/llama/Llama-3.2-3B-Instruct"
    "/mnt/sdb/models/Qwen/Qwen2.5-1.5B-Instruct"
)
# =============================

cleanup() {
    if [ -d "$TMP_WORK_DIR" ]; then
        echo -e "${YELLOW}🧹 正在清理临时目录: $TMP_WORK_DIR${NC}"
        rm -rf "$TMP_WORK_DIR"
    fi
}

trap cleanup EXIT ERR

cd "$WORK_DIR" || { echo -e "${RED}错误：无法进入目录 $WORK_DIR${NC}"; exit 1; }

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}错误：未找到 Python 脚本 $PYTHON_SCRIPT${NC}"
    exit 1
fi

source "$CONDA_PATH/etc/profile.d/conda.sh" || { echo -e "${RED}错误：无法加载 Conda${NC}"; exit 1; }
conda activate "$ENV_NAME" || { echo -e "${RED}错误：无法激活环境 $ENV_NAME${NC}"; exit 1; }
echo -e "${GREEN}✅ Conda 环境激活成功${NC}\n"

echo -e "${YELLOW}📂 正在复制数据到临时目录...${NC}"
cp -r "$DATA_ROOT"/* "$TMP_WORK_DIR/" || {
    echo -e "${RED}错误：复制数据失败${NC}"
    exit 1
}

mapfile -t INPUT_FILES < <(find "$TMP_WORK_DIR" -name "*.jsonl" | sort)
echo -e "${GREEN}📄 共发现 $((${#INPUT_FILES[@]})) 个 JSONL 文件${NC}"

for input_file in "${INPUT_FILES[@]}"; do
    filename=$(basename "$input_file" .jsonl)
    rel_path="${input_file#$TMP_WORK_DIR/}"
    echo -e "\n${GREEN}📄 正在处理文件: $rel_path${NC}"

    for MODEL_PATH in "${MODELS[@]}"; do
        MODEL_NAME=$(basename "$MODEL_PATH")
        FIELD_NAME="${MODEL_NAME}_generate"

        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        model_log="$LOG_DIR/${TIMESTAMP}_${MODEL_NAME}_${filename}.log"

        echo -e "${YELLOW}  🔧 模型: $MODEL_NAME${NC}"
        echo -e "     📝 日志: $model_log"

        python "$PYTHON_SCRIPT" \
            --input "$input_file" \
            --model_path "$MODEL_PATH" \
            --output "$input_file" \
            --field "$FIELD_NAME" >> "$model_log" 2>&1

        if [ $? -eq 0 ]; then
            echo -e "     ${GREEN}✅ 成功${NC}"
        else
            echo -e "     ${RED}❌ 失败${NC}"
        fi

        sleep 3
    done

    echo -e "${GREEN}✅ 文件 $rel_path 所有模型处理完成${NC}"
done

echo -e "${YELLOW}🚚 正在拷贝结果到 $RESULT_DIR ...${NC}"
cp -r "$TMP_WORK_DIR"/* "$RESULT_DIR/" || {
    echo -e "${RED}错误：拷贝结果失败${NC}"
    exit 1
}

echo -e "${GREEN}🎉 所有任务完成！结果已保存至: $RESULT_DIR ${NC}"

cleanup
