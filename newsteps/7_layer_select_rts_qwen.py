import os
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ---- 0. 核心修复：Patch torch.is_autocast_enabled 以支持任何参数 (如 device_type) ----
if hasattr(torch, "is_autocast_enabled"):
    _orig_is_autocast_enabled = torch.is_autocast_enabled
    def _patched_is_autocast_enabled(*args, **kwargs):
        # 忽略 device_type 等新版本参数，直接调用原函数
        return _orig_is_autocast_enabled()
    torch.is_autocast_enabled = _patched_is_autocast_enabled
    print("[PATCH] torch.is_autocast_enabled patched for compatibility.")


# ---- 1. RTS token id 读取 (支持从文本转换为 ID) ----

def load_rts_token_ids(path, processor=None):
    """
    递归搜索 JSON 中的所有值。
    如果发现字符串且提供了 processor，则将其转换为模型对应的 Token ID。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"RTS file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tokens_found = []
    ids_found = set()

    def extract_recursive(obj):
        if isinstance(obj, str):
            if obj.isdigit():
                ids_found.add(int(obj))
            else:
                tokens_found.append(obj)
        elif isinstance(obj, (int, float)):
            ids_found.add(int(obj))
        elif isinstance(obj, list):
            for item in obj:
                extract_recursive(item)
        elif isinstance(obj, dict):
            # 遍历 dictionary 的 values
            for v in obj.values():
                extract_recursive(v)

    extract_recursive(data)

    # 如果有收集到文本 token，利用 tokenizer 转换
    if tokens_found:
        if processor is None:
            print(f"[WARN] Found {len(tokens_found)} text tokens, but processor is None. Cannot convert to IDs.")
        else:
            print(f"[RTS] Converting {len(tokens_found)} text tokens to IDs...")
            for t in tokens_found:
                # encode 可能会将一个词拆分成多个 token，全部加入集合
                t_ids = processor.tokenizer.encode(t, add_special_tokens=False)
                for tid in t_ids:
                    ids_found.add(tid)

    if not ids_found:
        print(f"[DEBUG] JSON root keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        raise ValueError(f"No token ids found or could be converted in: {p}")

    print(f"[RTS] Successfully loaded {len(ids_found)} unique token ids.")
    return sorted(list(ids_found))


# ---- 2. 数据抽取工具 ----

def extract_prompt(rec):
    for key in ["prompt_used", "attack_text", "question", "text", "input", "prompt"]:
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def collect_safe_unsafe_samples(state_jsonl, max_refusal=32, max_unsafe=32):
    safe_samples, unsafe_samples = [], []
    with open(state_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            label = rec.get("state_label") or rec.get("label")
            if label not in ("refusal", "unsafe"): continue
            
            prompt = extract_prompt(rec)
            if not prompt: continue

            item = {"id": rec.get("id"), "prompt": prompt}
            if label == "refusal" and len(safe_samples) < max_refusal:
                safe_samples.append(item)
            elif label == "unsafe" and len(unsafe_samples) < max_unsafe:
                unsafe_samples.append(item)
            
            if len(safe_samples) >= max_refusal and len(unsafe_samples) >= max_unsafe:
                break

    print(f"[STATE] Collected {len(safe_samples)} safe (refusal) and {len(unsafe_samples)} unsafe samples.")
    if not safe_samples or not unsafe_samples:
        raise RuntimeError("Sample collection failed: safe or unsafe list is empty.")
    return safe_samples, unsafe_samples


# ---- 3. 推理逻辑 ----

def build_qwen_inputs(processor, prompts, device):
    messages_batch = []
    for text in prompts:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        messages_batch.append(chat)

    inputs = processor(text=messages_batch, images=None, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}

def compute_layer_scores_for_group(samples, processor, model, rts_token_ids, layer_indices, batch_size=4, desc="group"):
    device = next(model.parameters()).device
    rts_ids_tensor = torch.tensor(rts_token_ids, dtype=torch.long, device=device)
    layer_scores = {l: [] for l in layer_indices}

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size), desc=f"[FORWARD] {desc}"):
            batch = samples[i:i + batch_size]
            prompts = [s["prompt"] for s in batch]
            inputs = build_qwen_inputs(processor, prompts, device)
            
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            last_indices = inputs["attention_mask"].sum(dim=-1) - 1

            for b_idx in range(len(batch)):
                last_idx = int(last_indices[b_idx].item())
                for l in layer_indices:
                    h = hidden_states[l][b_idx, last_idx, :]
                    logits = model.lm_head(h)
                    norm = torch.norm(logits, p=2) + 1e-9
                    score = (logits[rts_ids_tensor].mean() / norm).item()
                    layer_scores[l].append(score)
    return layer_scores


# ---- 4. 汇总与输出 ----

def summarize_layers(safe_scores, unsafe_scores, out_csv):
    import statistics
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    rows = []
    for l in sorted(safe_scores.keys()):
        s_list, u_list = safe_scores[l], unsafe_scores.get(l, [])
        if not s_list or not u_list: continue

        s_mean, s_std = statistics.mean(s_list), (statistics.pstdev(s_list) if len(s_list)>1 else 0)
        u_mean, u_std = statistics.mean(u_list), (statistics.pstdev(u_list) if len(u_list)>1 else 0)
        delta = s_mean - u_mean

        rows.append({"layer": l, "safe_mean": s_mean, "safe_std": s_std, "unsafe_mean": u_mean, "unsafe_std": u_std, "delta": delta})

    delta_L = rows[-1]["delta"] if abs(rows[-1]["delta"]) > 1e-9 else (max(abs(r["delta"]) for r in rows) + 1e-9)
    for r in rows:
        r["lambda"] = r["delta"] / delta_L

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("layer,safe_mean,safe_std,unsafe_mean,unsafe_std,delta,lambda\n")
        for r in rows:
            f.write("{layer},{safe_mean:.6e},{safe_std:.6e},{unsafe_mean:.6e},{unsafe_std:.6e},{delta:.6e},{lambda:.6f}\n".format(**r))

    print(f"\n[TOP LAYERS]")
    for r in sorted(rows, key=lambda x: x["lambda"], reverse=True)[:10]:
        print(f"  Layer {r['layer']:2d}: Δ={r['delta']:+.3e}, λ={r['lambda']:+.3f}")
    return rows


# ---- 5. Main 主程序 ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_jsonl", type=str, default="data/text_state_collection.jsonl")
    parser.add_argument("--rts_json", type=str, default="work/rts/rts_final.json")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--max_refusal", type=int, default=32)
    parser.add_argument("--max_unsafe", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out_csv", type=str, default="work/layer_select/qwen_rts_layer_scan.csv")
    args = parser.parse_args()

    # 1. 加载 Processor (用于下一步解析 RTS)
    print(f"[LOAD] Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    # 2. 加载 RTS IDs (传入 processor)
    rts_ids = load_rts_token_ids(args.rts_json, processor=processor)

    # 3. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.torch_dtype) if args.torch_dtype != "auto" else "auto"
    print(f"[LOAD] Loading model: {args.model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map="auto"
    )

    # 4. 收集样本
    safe_samples, unsafe_samples = collect_safe_unsafe_samples(args.state_jsonl, args.max_refusal, args.max_unsafe)

    # 5. 确定层数
    tmp_inputs = build_qwen_inputs(processor, [safe_samples[0]["prompt"]], device)
    with torch.no_grad():
        tmp_out = model(**tmp_inputs, output_hidden_states=True)
    num_layers = len(tmp_out.hidden_states) - 1
    layer_indices = list(range(1, num_layers + 1))

    # 6. 计算得分
    safe_scores = compute_layer_scores_for_group(safe_samples, processor, model, rts_ids, layer_indices, args.batch_size, "safe")
    unsafe_scores = compute_layer_scores_for_group(unsafe_samples, processor, model, rts_ids, layer_indices, args.batch_size, "unsafe")

    # 7. 总结
    summarize_layers(safe_scores, unsafe_scores, args.out_csv)
    print("[DONE] Finished.")

if __name__ == "__main__":
    main()