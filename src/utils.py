import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import gc

class Qwen_LLM:
    def __init__(self, model_path="/mnt/sdb/models/Qwen/Qwen3-8B", device="cuda:0", torch_dtype=torch.bfloat16):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype

        print(f"正在加载模型：{model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )
        print(f"{model_path} 模型加载完成，运行在：{self.device}")

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=0.7, do_sample=False):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=1,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return response

    @torch.no_grad()
    def extract_hidden_states(self, texts, layer_range=None, mode="last_token"):
        if not texts:
            return {}

        num_layers = len(self.model.model.layers)
        layers = list(range(num_layers))

        if layer_range is not None:
            start, end = layer_range
            layers = [l for l in layers if start <= l < end]
        else:
            layer_range = (0, num_layers)

        print(f"正在提取第 {layer_range[0]} 到 {layer_range[1]} 层的隐藏状态...")

        # 存储结果：{idx: {layer: hidden_state}}
        hidden_states = {i: {} for i in range(len(texts))}

        for idx, text in tqdm(enumerate(texts), total=len(texts), desc="提取隐藏状态"):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(self.model.device)

            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            )
            hs = outputs.hidden_states  # tuple of (batch_size, seq_len, hidden_size)

            for layer in layers:
                h = hs[layer]  # shape: (1, seq_len, hidden_size)
                if mode == "last_token":
                    rep = h[0][-1]
                elif mode == "mean_pooling":
                    rep = h[0].mean(dim=0)
                else:
                    raise ValueError("mode 必须是 'last_token' 或 'mean_pooling'")

                hidden_states[idx][layer] = rep.cpu().to(torch.float32).numpy()

        return hidden_states

class Llama_LLM:
    def __init__(self, model_path="/mnt/sdb/models/llama/Llama-3.1-8B-Instruct/", device="auto", torch_dtype=torch.bfloat16):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype

        print(f"正在加载模型：{model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": self.torch_dtype},
            device_map=self.device,
        )

        print(f"{model_path} 模型加载完成，运行在：{self.device}")

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=0.7, do_sample=False):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        outputs = self.generator(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = outputs[0]["generated_text"][-1]["content"].strip()
        return response


    @torch.no_grad()
    def extract_hidden_states(self, texts, layer_range=None, mode="last_token"):
        if not texts:
            return {}

        num_layers = len(self.model.model.layers)
        layers = list(range(num_layers))

        # 解析 layer_range
        if layer_range is not None:
            start, end = layer_range
            layers = [l for l in layers if start <= l < end]
        else:
            layer_range = (0, num_layers)

        print(f"正在提取第 {layer_range[0]} 到 {layer_range[1]} 层的隐藏状态...")

        hidden_states = {i: {} for i in range(len(texts))}

        for idx, text in tqdm(enumerate(texts), total=len(texts), desc="提取隐藏状态"):
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=8192
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                )
                hs = outputs.hidden_states  # tuple of (batch, seq, dim)

            for layer in layers:
                h = hs[layer]  # shape: (1, seq_len, hidden_size)
                if mode == "last_token":
                    rep = h[0][-1]
                elif mode == "mean_pooling":
                    rep = h[0].mean(dim=0)
                else:
                    raise ValueError("mode 必须是 'last_token' 或 'mean_pooling'")

                hidden_states[idx][layer] = rep.cpu().to(torch.float32).numpy()

        return hidden_states