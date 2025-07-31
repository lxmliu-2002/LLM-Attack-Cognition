import os
import json
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def load_jsonl_labels(file_path):
    """读取 jsonl 文件，返回按 id 排序的 label 字典"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)
    data.sort(key=lambda x: x.get("id", 0))
    labels = {item["id"]: item["label"] for item in data}
    return labels


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





def main():
    
    parser = ArgumentParser(description="使用 t-SNE 可视化每层 hidden states，按 label 着色")
    parser.add_argument("--hs", type=str, required=True, help="hidden states .npy 文件路径")
    parser.add_argument("--jsonl", type=str, required=True, help="JSONL 标签文件路径（含 id 和 label）")
    parser.add_argument("--output_dir", type=str, required=True, help="保存图像的目录路径")

    args = parser.parse_args()

    labels_dict = load_jsonl_labels(args.jsonl)
    hiddens, ids = load_hidden_states(args.hs)

    ordered_labels = [labels_dict.get(idx, "unknown") for idx in ids]
    y = np.array([1 if lbl == "safe" else 0 for lbl in ordered_labels])

    model_name = os.path.basename(args.hs).split("_")[0]
    base_name = os.path.basename(args.hs)
    if base_name.endswith("_train.npy"):
        model_name = base_name[:-10]
    elif base_name.endswith("_eval.npy"):
        model_name = base_name[:-9]
    else:
        model_name = base_name.replace(".npy", "")

    os.makedirs(args.output_dir, exist_ok=True)

    layers = sorted(hiddens.keys())
    n_layers = len(layers)

    cols = (n_layers + 3) // 4
    rows = min(4, n_layers)
    if n_layers > 4:
        cols = (n_layers + 3) // 4
    else:
        cols = n_layers
        rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    print(f"🎨 开始 t-SNE 降维并绘图... 共 {n_layers} 层")

    for idx, layer in enumerate(layers):
        X = hiddens[layer]
        print(f"   → 第 {layer} 层: {X.shape}")

        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca')
        X_2d = tsne.fit_transform(X)

        ax = axes[idx]
        colors = ['red' if lbl == 0 else 'blue' for lbl in y]
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=5, alpha=0.7)
        ax.set_title(f"Layer {layer}", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_layers, len(axes)):
        fig.delaxes(axes[idx])

    # fig.suptitle(f"t-SNE Visualization of Hidden States - {model_name}", fontsize=16)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='safe'), Patch(facecolor='red', label='unsafe')]
    fig.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(args.output_dir, f"{model_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 图像已保存: {output_path}")


if __name__ == "__main__":
    main()
