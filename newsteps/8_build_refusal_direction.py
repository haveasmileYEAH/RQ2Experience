import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def patch_torch_autocast():
    """兼容老版本 torch.is_autocast_enabled(device_type) 的问题。"""
    try:
        sig = torch.is_autocast_enabled.__code__.co_argcount
        if sig == 0:
            _orig = torch.is_autocast_enabled

            def _patched(*args, **kwargs):
                return _orig()

            torch.is_autocast_enabled = _patched
    except Exception as e:
        print("[WARN] failed to patch torch.is_autocast_enabled:", repr(e))


def parse_layers(arg, num_hidden_layers: int):
    """
    解析 --layers 参数：
      - "auto"  ->  [1..num_hidden_layers]
      - "13,27,28" -> [13,27,28]
      - "8-28"  ->  [8,9,...,28]
      - 混合："8-12,27,28"
    全部用 1-based 索引（和你之前 λ 报表一致）。
    """
    if arg == "auto":
        return list(range(1, num_hidden_layers + 1))

    layers = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            a = int(a)
            b = int(b)
            if a > b:
                a, b = b, a
            layers.extend(range(a, b + 1))
        else:
            layers.append(int(part))
    # 去重 + 排序
    layers = sorted(set(layers))
    # 限制在 [1, num_hidden_layers]
    layers = [l for l in layers if 1 <= l <= num_hidden_layers]
    return layers


def read_states(path, max_samples=None):
    """
    读取 7_collect_refusal_vs_unsafe_states.py 的输出：
    要求每行至少包含:
      - 'prompt' 或 'text'
      - 'state_label' ∈ {'refusal', 'unsafe', 'other'}
    这里只保留 'refusal' & 'unsafe' 两类。
    """
    path = Path(path)
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            label = r.get("state_label")
            if label not in ["refusal", "unsafe"]:
                continue

            prompt = (
                r.get("prompt")
                or r.get("text")
                or r.get("query")
                or r.get("attack_text")
            )
            if not isinstance(prompt, str) or not prompt.strip():
                continue

            data.append(
                {
                    "prompt": prompt.strip(),
                    "state_label": label,
                    "id": r.get("id"),
                }
            )
            if max_samples is not None and len(data) >= max_samples:
                break

    n_r = sum(1 for x in data if x["state_label"] == "refusal")
    n_u = sum(1 for x in data if x["state_label"] == "unsafe")
    print(f"[LOAD] {path} rows={len(data)}, refusal={n_r}, unsafe={n_u}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state_jsonl",
        type=str,
        required=True,
        help="7_collect_refusal_vs_unsafe_states.py 的输出，例如 data/text_state_collection.jsonl",
    )
    ap.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    ap.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    ap.add_argument(
        "--layers",
        type=str,
        default="auto",
        help='使用哪些层构造方向，如 "auto" 或 "13,27,28" 或 "8-28"',
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="可选：只用前 N 条样本构造方向（调试用）",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="work/refusal_direction",
    )
    args = ap.parse_args()

    patch_torch_autocast()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[CFG] device={device}, dtype={dtype}")
    print(f"[CFG] model_id={args.model_id}")

    # 1) 模型
    print("[LOAD] processor & model...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    # Qwen2_5_VL 使用 num_layers
    num_layers = getattr(model.config, "num_layers", getattr(model.config, "num_hidden_layers", 28))
    print(f"[INFO] num_hidden_layers={num_layers}")

    # 2) 层选择
    layer_ids = parse_layers(args.layers, num_layers)
    print(f"[INFO] using layers: {layer_ids}")

    # 3) 数据
    data = read_states(args.state_jsonl, max_samples=args.max_samples)
    if not data:
        raise RuntimeError("No valid states found in state_jsonl.")

    # 4) 为每一层维护 R/U 的累积向量 + 计数
    R_sum = {}  # layer -> tensor(D,)
    U_sum = {}
    R_cnt = {l: 0 for l in layer_ids}
    U_cnt = {l: 0 for l in layer_ids}

    with torch.inference_mode():
        for row in tqdm(data, desc="Step8: collect layer-wise means"):
            prompt = row["prompt"]
            label = row["state_label"]  # 'refusal' or 'unsafe'

            inputs = processor(
                text=prompt,
                images=None,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states  # tuple length = num_layers+1

            for lid in layer_ids:
                # hidden_states[lid] is (B, T, D); B=1
                h = hidden_states[lid][:, -1, :]  # 最后一个 token 的表示 (1,D)
                h = F.normalize(h, dim=-1)  # 单位化，防止某些样本能量过大
                h = h.squeeze(0)  # (D,)

                if label == "refusal":
                    if lid not in R_sum:
                        R_sum[lid] = torch.zeros_like(h)
                    R_sum[lid] += h
                    R_cnt[lid] += 1
                elif label == "unsafe":
                    if lid not in U_sum:
                        U_sum[lid] = torch.zeros_like(h)
                    U_sum[lid] += h
                    U_cnt[lid] += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model_id": args.model_id,
        "state_jsonl": args.state_jsonl,
        "layers": layer_ids,
        "torch_dtype": args.torch_dtype,
        "num_samples": len(data),
        "refusal_counts": R_cnt,
        "unsafe_counts": U_cnt,
        "note": "direction_l = normalize( mean_R(l) - mean_U(l) ), using last-token hidden state with per-sample L2 norm.",
    }

    # 5) 计算每一层的 direction 并保存
    saved_layers = []
    for lid in layer_ids:
        rc = R_cnt.get(lid, 0)
        uc = U_cnt.get(lid, 0)
        if rc == 0 or uc == 0:
            print(f"[WARN] layer {lid}: rc={rc}, uc={uc}, skip.")
            continue

        mu_R = R_sum[lid] / rc
        mu_U = U_sum[lid] / uc
        d = mu_R - mu_U
        d = F.normalize(d, dim=-1)
        # 先转为 float32 (float())，再转为 numpy
        d_np = d.detach().cpu().float().numpy()

        fname = out_dir / f"layer{lid:02d}.npy"
        np.save(fname, d_np)
        saved_layers.append(lid)
        print(f"[SAVE] layer {lid}: direction -> {fname}, dim={d_np.shape}")

    meta["saved_layers"] = saved_layers

    meta_path = out_dir / "refusal_direction_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] meta written to {meta_path}")
    print(f"[SUMMARY] saved layers: {saved_layers}")


if __name__ == "__main__":
    main()
