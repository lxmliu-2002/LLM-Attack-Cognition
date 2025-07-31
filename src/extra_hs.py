# python extra_hs.py --input /mnt/sdc/user_workspace/liuxiuming/Projects/ant/LLM-Attack-Cognition/data/mlp/train/train.jsonl --model_path /mnt/sdb/models/Qwen/Qwen3-4B 

import os
import json
import argparse
import numpy as np
from pathlib import Path
import torch

from utils import Qwen_LLM, Llama_LLM


def detect_model_type(model_path):
    """
    根据模型路径判断是 Qwen 还是 Llama 模型
    """
    model_path = model_path.lower()
    if "qwen" in model_path:
        return "qwen"
    elif "llama" in model_path:
        return "llama"
    else:
        raise ValueError(f"无法识别模型类型: {model_path}")


def load_jsonl(file_path):
    """读取 jsonl 文件，返回 prompt 列表和原始数据"""
    prompts = []
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                prompts.append(item.get("prompt", ""))
                data.append(item)
    return prompts, data


def main():
    parser = argparse.ArgumentParser(description="提取 JSONL 中 prompt 的隐藏状态")
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--output_dir", type=str, required=True, help="保存路径")

    args = parser.parse_args()

    input_file = args.input
    model_path = args.model_path
    output_dir = args.output_dir

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    model_type = detect_model_type(model_path)
    print(f"🔍 检测到模型类型: {model_type.upper()}")

    if model_type == "qwen":
        llm = Qwen_LLM(model_path=model_path, device="auto", torch_dtype=torch.bfloat16)
    else:
        llm = Llama_LLM(model_path=model_path, device="auto", torch_dtype=torch.bfloat16)

    model_name = Path(model_path).name
    model_name = model_name.replace("-", "_").replace(".", "_")

    print(f"📄 正在加载 {input_file} ...")
    prompts, raw_data = load_jsonl(input_file)

    print(f"✅ 共加载 {len(prompts)} 条 prompt")

    print("🧠 正在提取隐藏状态...")
    hidden_states = llm.extract_hidden_states(
        texts=prompts,
        layer_range=None,  # None 代表所有层
        mode="last_token"
    )

    os.makedirs(output_dir, exist_ok=True)

    input_filename = Path(input_file).stem
    output_file = f"{model_name}_{input_filename}.npy"
    output_path = os.path.join(output_dir, output_file)

    print(f"💾 正在保存隐藏状态到: {output_path}")
    np.save(output_path, hidden_states)

    print(f"🎉 提取完成！共 {len(prompts)} 条数据，保存至:\n  {output_path}")



if __name__ == "__main__":
    main()