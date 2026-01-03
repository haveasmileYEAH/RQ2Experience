import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from sklearn.metrics import roc_auc_score, average_precision_score

# ------------------------------------------------
# 兼容 Qwen2.5-VL 的 autocast bug（和你 step5 用的一样思路）
# ------------------------------------------------
def patch_torch_autocast():
    import torch
    import transformers.utils.generic

    # 1. 保存原始函数
    _orig_is_autocast = torch.is_autocast_enabled

    # 2. 定义兼容性函数
    def patched_is_autocast(device_type=None):
        # 无论传什么参数，都只调用原始的无参函数
        return _orig_is_autocast()

    # 3. 全局替换
    torch.is_autocast_enabled = patched_is_autocast

    # 4. 关键：直接替换 transformers 报错位置的引用
    # 报错信息显示是 transformers.utils.generic 调用的
    transformers.utils.generic.torch.is_autocast_enabled = patched_is_autocast
    
    # 5. 额外保险：替换模型内部可能存在的局部引用
    try:
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_mod
        qwen_mod.torch.is_autocast_enabled = patched_is_autocast
    except ImportError:
        pass

    print("[INFO] Monkey patch for is_autocast_enabled applied successfully.")


# ------------------------------------------------
# 数据集读取：text_attack_xstest.jsonl
# 假设每行至少有一个 text 字段和一个二分类 label 字段
# 你如果字段名不一样，只改这里就行
# ------------------------------------------------
def load_text_binary_dataset(path, max_samples=None):
    ds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            # 1) prompt / text 字段
            prompt = (
                r.get("prompt")
                or r.get("text")
                or r.get("question")
                or r.get("xs_query")
            )
            if not isinstance(prompt, str):
                continue

            # 2) label 字段：1 = harmful / 0 = benign
            #    按你自己的 jsonl 实际字段调整这一段即可
# 2) label 字段：1 = harmful / 0 = benign
            raw_label = (
                r.get("label")
                if r.get("label") is not None
                else r.get("is_harmful")
            )
            
            # 增加对字符串标签的兼容性映射
            if isinstance(raw_label, str):
                raw_label = raw_label.lower().strip()
                if raw_label in ["1", "harmful", "unsafe", "attack"]:
                    label = 1
                elif raw_label in ["0", "benign", "safe", "normal"]:
                    label = 0
                else:
                    # 如果还是没匹配到，尝试看 xs_label
                    xs_label = str(r.get("xs_label", "")).lower()
                    if "harmful" in xs_label or "unsafe" in xs_label:
                        label = 1
                    elif "benign" in xs_label or "safe" in xs_label:
                        label = 0
                    else:
                        continue # 还是识别不了就跳过
            else:
                # 如果已经是数字或 None
                label = raw_label

            if label is None:
                continue

            label = int(label)

            ds.append({"prompt": prompt, "label": label})
            if max_samples is not None and len(ds) >= max_samples:
                break

    print(f"[LOAD] dataset from {path}, samples={len(ds)}")
    return ds


# ------------------------------------------------
# 加载 refusal direction（你在 step8 里生成的 layerXX.npy）
# ------------------------------------------------
def load_refusal_directions(refusal_dir, layer_ids):
    refusal_dir = Path(refusal_dir)
    dirs = {}
    for lid in layer_ids:
        # 假定文件名形如 layer28.npy / layer13.npy
        cand_names = [
            f"layer{lid:02d}.npy",
            f"layer_{lid}.npy",
            f"layer{lid}.npy",
        ]
        found = None
        for name in cand_names:
            p = refusal_dir / name
            if p.exists():
                found = p
                break
        if found is None:
            raise FileNotFoundError(
                f"Cannot find direction file for layer {lid} in {refusal_dir}"
            )
        vec = np.load(found)
        dirs[lid] = torch.from_numpy(vec).float()
        print(f"[LOAD] direction layer {lid} from {found}, dim={dirs[lid].shape}")
    return dirs


# ------------------------------------------------
# 对单个样本计算“多层拒绝得分”
# 这里使用：每层用 cos_sim(h_l, d_l)，然后在层轴上做积分 / 平均
# ------------------------------------------------
def score_sample_multilayer(model, processor, directions, layers, prompt, device):
    # Qwen2.5-VL 的最简单用法：直接 text → input_ids
    inputs = processor(
        text=prompt,
        images=None,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states  # tuple(len = n_layers+1)

    layer_scores = []
    x_coords = []

    for lid in layers:
        # hidden_states[lid] shape: (B, T, D), B=1
        h = hidden_states[lid][:, -1, :]  # 取最后一个 token 的 hidden state
        d = directions[lid].to(device)
        if d.dim() == 1:
            d = d.unsqueeze(0)  # (1, D)

        # cos_sim 作为该层的“refusal 强度”
        cs = F.cosine_similarity(h, d, dim=-1)  # (1,)
        layer_scores.append(cs.item())
        x_coords.append(float(lid))

    if len(layer_scores) == 0:
        return 0.0

    # 方式 A：简单平均
    # score = float(np.mean(layer_scores))

    # 方式 B：仿 HiddenDetect，按层号做一维积分（再做一个归一化）
    xs = np.array(x_coords, dtype=np.float32)
    ys = np.array(layer_scores, dtype=np.float32)
    if len(xs) > 1:
        integral = float(np.trapz(ys, xs))
        # 归一化到 [s,e] 区间长度上，避免不同层数导致尺度不同
        score = integral / (xs[-1] - xs[0])
    else:
        score = ys[0]

    return score


# ------------------------------------------------
# 主流程：加载模型 + directions + 数据集 → 计算 AUROC / AUPRC
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--text_attack_jsonl",
        type=str,
        required=True,
        help="如 data/text_attack_xstest.jsonl",
    )
    parser.add_argument(
        "--refusal_direction_dir",
        type=str,
        required=True,
        help="例如 work/refusal_direction",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="13,27,28",
        help="使用哪些层进行多层积分，逗号分隔，1-based，如 13,27,28",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="可选：只用前 N 条样本做快速测试",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="work/refusal_direction/qwen_multilayer_eval.csv",
    )

    args = parser.parse_args()
    patch_torch_autocast()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[CFG] device={device}, dtype={dtype}")

    # 1) 模型 & processor
    print(f"[LOAD] loading Qwen2.5-VL model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    # 2) directions
    layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]
    layer_ids = sorted(layer_ids)
    directions = load_refusal_directions(args.refusal_direction_dir, layer_ids)

    # 3) 数据集
    dataset = load_text_binary_dataset(args.text_attack_jsonl, max_samples=args.max_samples)

    # 4) 逐样本评分
    all_labels = []
    all_scores = []

    for sample in tqdm(dataset, desc="Eval multi-layer refusal direction"):
        prompt = sample["prompt"]
        label = int(sample["label"])

        score = score_sample_multilayer(
            model=model,
            processor=processor,
            directions=directions,
            layers=layer_ids,
            prompt=prompt,
            device=device,
        )

        all_labels.append(label)
        all_scores.append(score)

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # 5) 指标
    auroc = roc_auc_score(all_labels, all_scores)
    auprc = average_precision_score(all_labels, all_scores)

    print("\n[RESULT]")
    print(f"  #samples: {len(all_labels)}")
    print(f"  AUROC:    {auroc:.4f}")
    print(f"  AUPRC:    {auprc:.4f}")

    # 6) 导出原始 scores（方便你后面画图 / 诊断）
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("score,label\n")
        for s, y in zip(all_scores, all_labels):
            f.write(f"{s:.8f},{y}\n")

    print(f"[OK] raw scores written to {out_path}")


if __name__ == "__main__":
    main()
