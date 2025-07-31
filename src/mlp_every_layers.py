import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tabulate import tabulate


def load_jsonl_labels(file_path):
    """读取 jsonl 文件，返回按 id 排序的 label 列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)

    data.sort(key=lambda x: x.get("id", 0))
    ids = [item["id"] for item in data]
    labels = [1 if item["label"] == "safe" else 0 for item in data]
    return np.array(labels), ids


def load_hidden_states(np_path):
    """加载 .npy 文件中的 hidden states"""
    print(f"🧠 正在加载: {np_path}")
    data_raw = np.load(np_path, allow_pickle=True)
    
    if data_raw.ndim != 0:
        raise ValueError(f"Expected 0-dim ndarray, got shape {data_raw.shape}")
    
    data = data_raw.item()

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")

    ids = sorted(data.keys())
    if len(ids) == 0:
        raise ValueError("No samples found in hidden states file")

    sample_data = data[ids[0]]
    if not isinstance(sample_data, dict):
        raise TypeError(f"Each sample should be a dict of layers, got {type(sample_data)}")
    
    layers = sorted(sample_data.keys())
    print(f"   → 检测到 {len(ids)} 个样本，{len(layers)} 层")

    all_hiddens = {}
    for layer in layers:
        features = []
        for idx in ids:
            if layer not in data[idx]:
                raise KeyError(f"Layer {layer} not found in sample {idx}")
            layer_feat = data[idx][layer]
            if isinstance(layer_feat, list):
                if len(layer_feat) == 0:
                    raise ValueError(f"Empty list for sample {idx}, layer {layer}")
                feat = layer_feat[0]
            else:
                feat = layer_feat
            if not isinstance(feat, np.ndarray):
                raise TypeError(f"Feature must be np.ndarray, got {type(feat)}")
            if feat.ndim != 1:
                print(f"⚠️ 展平非一维向量: sample {idx}, layer {layer}, shape {feat.shape}")
                feat = feat.flatten()

            features.append(feat)
        
        features = np.array(features)
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D after stacking, got shape {features.shape}")
        
        all_hiddens[layer] = features

    return all_hiddens, ids


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x.squeeze()


class HiddenStateDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def compute_tpr_at_fpr(y_true, y_scores, fpr_levels=[0.001, 0.01, 0.05]):
    """计算 TPR @ FPR"""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    tpr_at = []
    for target in fpr_levels:
        tpr_values = tpr[fpr <= target]
        tpr_at.append(tpr_values[-1] if len(tpr_values) > 0 else 0.0)
    return tpr_at


def train_and_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    print("📄 加载训练集标签...")
    train_labels, train_ids = load_jsonl_labels(args.train_jsonl)
    print("📄 加载验证集标签...")
    eval_labels, eval_ids = load_jsonl_labels(args.eval_jsonl)

    print("🧠 加载训练集 hidden states...")
    train_hiddens, train_hs_ids = load_hidden_states(args.train_hs)
    print("🧠 加载验证集 hidden states...")
    eval_hiddens, eval_hs_ids = load_hidden_states(args.eval_hs)

    assert train_ids == train_hs_ids, "训练集 label 和 hidden state 的 id 不一致"
    assert eval_ids == eval_hs_ids, "验证集 label 和 hidden state 的 id 不一致"
    assert len(train_labels) == len(train_ids), "label 数量不匹配"
    assert len(eval_labels) == len(eval_ids), "label 数量不匹配"

    train_basename = os.path.basename(args.train_hs)
    model_name = "_".join(train_basename.split("_")[:-1])
    layers = sorted(train_hiddens.keys())
    print(f"🔍 检测到模型: {model_name}, 共 {len(layers)} 层\n")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"📁 输出模型将保存到: {args.output_dir}")

    results_table = []

    for layer in layers:
        print(f"🔁 训练 {model_name} 第 {layer} 层 MLP 分类器...")

        X_train = train_hiddens[layer]  # (N_train, D)
        X_eval = eval_hiddens[layer]    # (N_eval, D)
        y_train = train_labels
        y_eval = eval_labels

        input_dim = X_train.shape[1]

        mean = X_train.mean(0, keepdims=True)
        std = X_train.std(0, keepdims=True) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_eval_norm = (X_eval - mean) / std

        train_dataset = HiddenStateDataset(X_train_norm, y_train)
        eval_dataset = HiddenStateDataset(X_eval_norm, y_eval)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

        model = MLP(input_dim).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        for epoch in range(20):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

        if args.output_dir:
            model_filename = f"{model_name}_{layer}.pt"
            model_save_path = os.path.join(args.output_dir, model_filename)
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 保存模型: {model_save_path}")

        model.eval()
        eval_preds = []
        eval_scores = []
        with torch.no_grad():
            for x, y in eval_loader:
                x = x.to(device)
                y_score = model(x).cpu().numpy()
                y_pred = (y_score >= 0.5).astype(int)
                eval_preds.extend(y_pred)
                eval_scores.extend(y_score)

        acc = accuracy_score(y_eval, eval_preds)
        prec = precision_score(y_eval, eval_preds)
        rec = recall_score(y_eval, eval_preds)
        f1 = f1_score(y_eval, eval_preds)
        auc = roc_auc_score(y_eval, eval_scores)
        tpr_01, tpr_1, tpr_5 = compute_tpr_at_fpr(y_eval, eval_scores)

        results_table.append([
            f"{model_name}_L{layer}",
            f"{len(train_labels)}",
            f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}",
            f"{auc:.4f}", f"{tpr_01:.4f}", f"{tpr_1:.4f}", f"{tpr_5:.4f}"
        ])

    headers = [
        "Model-Layer", "Train Size",
        "Acc", "Prec", "Rec", "F1",
        "AUC", "TPR@0.1%FPR", "TPR@1%FPR", "TPR@5%FPR"
    ]
    print("\n" + tabulate(results_table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 MLP 分类器并打印性能结果")
    parser.add_argument("--train_hs", type=str, required=True, help="训练集 hidden states .npy 文件路径")
    parser.add_argument("--eval_hs", type=str, required=True, help="验证集 hidden states .npy 文件路径")
    parser.add_argument("--train_jsonl", type=str, required=True, help="训练集 JSONL 文件路径")
    parser.add_argument("--eval_jsonl", type=str, required=True, help="验证集 JSONL 文件路径")
    parser.add_argument("--output_dir", type=str, default=None, help="保存训练好的 MLP 模型的目录（可选）")

    args = parser.parse_args()
    train_and_evaluate(args)