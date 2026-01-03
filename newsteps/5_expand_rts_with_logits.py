#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 5: 基于 Qwen2.5-VL 的 logits 扩展 RTS（拒绝表述集合）

输入：
  - 文本拒绝池:  data/qwen_refusals_text_xstest.jsonl
  - 图文拒绝池:  data/qwen_refusals_vision_mmsb.jsonl
  - 手工 RTS:   work/rts/rts_final_manual.json

输出：
  - 扩展 RTS:    work/rts/rts_expanded.json
  - 新 token 统计: work/rts/rts_new_tokens_from_logits.csv
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch

# ---- Patch torch.is_autocast_enabled 兼容 Qwen2.5-VL ----
# 目的：让 transformers.utils.generic.maybe_autocast 调用
# torch.is_autocast_enabled(device_type=...) 不再报 TypeError
try:
    if not hasattr(torch, "_orig_is_autocast_enabled"):
        torch._orig_is_autocast_enabled = torch.is_autocast_enabled

        def _patched_is_autocast_enabled(*args, **kwargs):
            # 忽略传入的 device_type 等参数，直接调用原始的无参版本
            return torch._orig_is_autocast_enabled()

        torch.is_autocast_enabled = _patched_is_autocast_enabled
        print("[INFO] patched torch.is_autocast_enabled to accept *args/**kwargs")
except Exception as e:
    print("[WARN] could not patch torch.is_autocast_enabled:", repr(e))
# --------------------------------------------------------
from tqdm import tqdm

# ========= 兼容补丁：修复老版本 torch.is_autocast_enabled 不接受 device_type 参数 =========
try:
    import inspect

    sig = inspect.signature(torch.is_autocast_enabled)
    if len(sig.parameters) == 0:
        _orig_is_autocast_enabled = torch.is_autocast_enabled

        def _is_autocast_enabled_compat(device_type: Optional[str] = None):
            # 忽略 device_type，直接调用原始版本
            return _orig_is_autocast_enabled()

        torch.is_autocast_enabled = _is_autocast_enabled_compat  # type: ignore
        print("[PATCH] torch.is_autocast_enabled patched to accept device_type (ignored).")
except Exception as e:
    print("[WARN] failed to patch torch.is_autocast_enabled:", repr(e))

from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration


# -------------------------
# 基础 IO & 判断函数
# -------------------------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def is_refusal_row(row: Dict[str, Any]) -> bool:
    if "is_refusal" in row:
        return bool(row.get("is_refusal"))
    if "is_refusal_rule" in row:
        return bool(row.get("is_refusal_rule"))
    return False


def get_response_text(row: Dict[str, Any]) -> str:
    for k in ["model_response", "response", "output", "answer"]:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# -------------------------
# 模型加载 & RTS ID 集
# -------------------------
def load_qwen_vl_text_model(
    model_id: str,
    torch_dtype: str = "bfloat16",
):
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(torch_dtype, torch.bfloat16)

    print(f"[LOAD] loading tokenizer & Qwen2.5-VL model: {model_id} (dtype={torch_dtype})")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def build_rts_token_id_set(
    manual_tokens: List[str],
    tokenizer,
) -> Tuple[Set[int], Dict[int, str]]:
    """
    把手工 RTS 里的词转换为 BPE token id 集合，用于后续过滤“已知 token”。
    """
    id_set: Set[int] = set()
    id2tok: Dict[int, str] = {}

    for t in manual_tokens:
        if not t:
            continue
        # 前面加一个空格以匹配 GPT/BPE 的空格前缀
        enc = tokenizer(
            " " + t,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = enc["input_ids"]
        for tid in ids:
            id_set.add(tid)
            if tid not in id2tok:
                id2tok[tid] = tokenizer.convert_ids_to_tokens([tid])[0]
    return id_set, id2tok


# -------------------------
# Layer 解析
# -------------------------
def parse_layers(arg: str, num_layers: int) -> List[int]:
    """
    支持形式：
      - "all"  -> [1..num_layers]
      - "last" -> [num_layers]
      - "18,20,22"
      - "10-20" (含头尾)
    注意：hidden_states[0] 是 embedding，从 1 开始是第 1 层。
    """
    if arg.lower() == "all":
        return list(range(1, num_layers + 1))
    if arg.lower() == "last":
        return [num_layers]

    if "," in arg:
        layers = []
        for x in arg.split(","):
            x = x.strip()
            if not x:
                continue
            layers.append(int(x))
        return layers

    if "-" in arg:
        a, b = arg.split("-", 1)
        start = int(a.strip())
        end = int(b.strip())
        return list(range(start, end + 1))

    # 单个数字
    return [int(arg)]


# -------------------------
# 主逻辑：从一个 jsonl split 中收集新 token
# -------------------------
def collect_new_tokens_from_split(
    split_name: str,
    jsonl_path: str,
    tokenizer,
    model,
    rts_id_set: Set[int],
    max_samples: int,
    layer_indices: List[int],
    top_k_per_layer: int,
    pool_last_n_tokens: int,
) -> Tuple[Dict[int, Dict[str, Any]], int]:
    """
    返回：
      - new_token_stats: token_id -> {count, layers, splits}
      - used_samples: 实际使用的拒绝样本数
    """
    device = next(model.parameters()).device
    new_token_stats: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "layers": set(), "splits": set()}
    )

    used_samples = 0
    total_rows = 0

    pbar = tqdm(
        read_jsonl(jsonl_path),
        desc=f"[{split_name}] logits expansion",
        unit="resp",
    )

    for row in pbar:
        total_rows += 1
        if not is_refusal_row(row):
            continue
        text = get_response_text(row)
        if not text:
            continue

        used_samples += 1
        if used_samples > max_samples:
            break

        # 编码文本
        enc = tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        seq_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states  # tuple, len = num_layers+1

        # 池化最后 pool_last_n_tokens 个 token 的 hidden state
        start_idx = max(0, seq_len - pool_last_n_tokens)
        end_idx = seq_len  # 不含
        token_range = range(start_idx, end_idx)

        for l in layer_indices:
            if not (0 <= l < len(hidden_states)):
                # 注意：hidden_states[0] 是 embedding；一般从 1..num_layers
                continue
            h = hidden_states[l][0, start_idx:end_idx, :]  # [T, H]
            if h.numel() == 0:
                continue
            h_mean = h.mean(dim=0)  # [H]

            # 通过 lm_head 得到 logits
            logits = model.lm_head(h_mean)  # [V]
            topk = torch.topk(logits, k=top_k_per_layer)
            top_ids = topk.indices.tolist()

            for tid in top_ids:
                if tid in rts_id_set:
                    # 已经在手工 RTS 里的 token，不算作“新 token”
                    continue
                stat = new_token_stats[tid]
                stat["count"] += 1
                stat["layers"].add(int(l))
                stat["splits"].add(split_name)

        pbar.set_postfix(
            used=used_samples,
            total_rows=total_rows,
            uniq_new=len(new_token_stats),
        )

    return new_token_stats, used_samples


# -------------------------
# CSV 输出
# -------------------------
def write_new_tokens_csv(
    out_csv: str,
    new_stats: Dict[int, Dict[str, Any]],
    tokenizer,
):
    out_p = Path(out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "token_id",
        "token",
        "raw_token",
        "count",
        "num_layers",
        "layers",
        "splits",
    ]

    with out_p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tid, stat in sorted(
            new_stats.items(),
            key=lambda kv: kv[1]["count"],
            reverse=True,
        ):
            tok = tokenizer.convert_ids_to_tokens([tid])[0]
            clean = tok.replace("▁", " ").strip()
            layers = sorted(list(stat["layers"]))
            splits = sorted(list(stat["splits"]))
            writer.writerow(
                {
                    "token_id": tid,
                    "token": clean,
                    "raw_token": tok,
                    "count": stat["count"],
                    "num_layers": len(layers),
                    "layers": ",".join(str(x) for x in layers),
                    "splits": ",".join(splits),
                }
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refusal_text_jsonl", type=str, required=True,
                    help="qwen_refusals_text_xstest.jsonl")
    ap.add_argument("--refusal_vision_jsonl", type=str, required=True,
                    help="qwen_refusals_vision_mmsb.jsonl")
    ap.add_argument("--rts_manual_json", type=str, required=True,
                    help="work/rts/rts_final_manual.json")
    ap.add_argument("--model_id", type=str, required=True,
                    help="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--torch_dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max_text_samples", type=int, default=40,
                    help="每个 split 最多使用多少拒绝样本（文本）")
    ap.add_argument("--max_vision_samples", type=int, default=40,
                    help="每个 split 最多使用多少拒绝样本（图文）")
    ap.add_argument("--layers", type=str, default="last",
                    help='如 "last", "all", "18,20,22", "10-20"')
    ap.add_argument("--top_k_per_layer", type=int, default=10,
                    help="每层取多少 logits top-k token")
    ap.add_argument("--pool_last_n_tokens", type=int, default=32,
                    help="对回答末尾多少个 token 做平均池化")
    ap.add_argument("--out_json", type=str, required=True,
                    help="扩展后的 RTS JSON 输出路径")
    ap.add_argument("--out_csv_new", type=str, required=True,
                    help="新 token 统计 CSV 输出路径")

    args = ap.parse_args()

    # 读取手工 RTS
    manual = json.loads(Path(args.rts_manual_json).read_text(encoding="utf-8"))
    manual_tokens: List[str] = list(manual.get("tokens", []))
    manual_tokens = [t.strip() for t in manual_tokens if t and t.strip()]

    # 加载模型 & tokenizer
    tokenizer, model = load_qwen_vl_text_model(
        args.model_id,
        torch_dtype=args.torch_dtype,
    )
    # ---- robust way to get num_layers for Qwen2.5-VL ----
    cfg = model.config
    
    # 1) 尝试直接读顶层字段
    num_layers = getattr(cfg, "num_hidden_layers", None)
    
    # 2) 如果没有，再尝试从 text_config 里读
    if num_layers is None and hasattr(cfg, "text_config"):
        num_layers = getattr(cfg.text_config, "num_hidden_layers", None)
    
    # 3) 如果还是 None，就从实际模型层结构推断
    if num_layers is None:
        try:
            # Qwen2.5-VLForConditionalGeneration 通常有 model.layers 或 model.model.layers
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                num_layers = len(model.model.layers)
            elif hasattr(model, "layers"):
                num_layers = len(model.layers)
            else:
                raise RuntimeError("Cannot infer num_layers from model structure")
        except Exception as e:
            raise RuntimeError(
                f"Cannot determine num_hidden_layers from config; "
                f"config attrs={dir(cfg)}"
            ) from e
    
    print(f"[INFO] detected num_layers = {num_layers}")
# ------------------------------------------------------

    layer_indices = parse_layers(args.layers, num_layers)
    print(f"[INFO] model num_hidden_layers = {num_layers}, using layers: {layer_indices}")

    # RTS id 集
    rts_id_set, rts_id2tok = build_rts_token_id_set(manual_tokens, tokenizer)
    print(f"[INFO] manual RTS tokens: {len(manual_tokens)}, RTS token ids: {len(rts_id_set)}")

    # 收集新 token（文本 split）
    new_text_stats, used_text = collect_new_tokens_from_split(
        split_name="text",
        jsonl_path=args.refusal_text_jsonl,
        tokenizer=tokenizer,
        model=model,
        rts_id_set=rts_id_set,
        max_samples=args.max_text_samples,
        layer_indices=layer_indices,
        top_k_per_layer=args.top_k_per_layer,
        pool_last_n_tokens=args.pool_last_n_tokens,
    )

    # 收集新 token（图文 split）
    new_vision_stats, used_vision = collect_new_tokens_from_split(
        split_name="vision",
        jsonl_path=args.refusal_vision_jsonl,
        tokenizer=tokenizer,
        model=model,
        rts_id_set=rts_id_set,
        max_samples=args.max_vision_samples,
        layer_indices=layer_indices,
        top_k_per_layer=args.top_k_per_layer,
        pool_last_n_tokens=args.pool_last_n_tokens,
    )

    # 合并两个 split 的新 token 统计
    merged_stats: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "layers": set(), "splits": set()}
    )
    for src in [new_text_stats, new_vision_stats]:
        for tid, st in src.items():
            m = merged_stats[tid]
            m["count"] += st["count"]
            m["layers"].update(st["layers"])
            m["splits"].update(st["splits"])

    # 生成 new_tokens & new_tokens_info
    new_tokens_info = []
    new_tokens_strs = []
    for tid, st in sorted(
        merged_stats.items(),
        key=lambda kv: kv[1]["count"],
        reverse=True,
    ):
        tok = tokenizer.convert_ids_to_tokens([tid])[0]
        clean = tok.replace("▁", " ").strip()
        if not clean:
            continue
        info = {
            "token_id": tid,
            "token": clean,
            "raw_token": tok,
            "count": int(st["count"]),
            "layers": sorted(list(st["layers"])),
            "splits": sorted(list(st["splits"])),
        }
        new_tokens_info.append(info)
        new_tokens_strs.append(clean)

    # 写 CSV
    write_new_tokens_csv(
        args.out_csv_new,
        merged_stats,
        tokenizer,
    )

    # 最终 RTS = 手工 RTS ∪ 新 token
    final_tokens = manual_tokens + [t for t in new_tokens_strs if t not in manual_tokens]

    out_obj = {
        "manual_tokens": manual_tokens,
        "new_tokens": new_tokens_strs,
        "tokens": final_tokens,
        "new_tokens_info": new_tokens_info,
        "meta": {
            "model_id": args.model_id,
            "torch_dtype": args.torch_dtype,
            "refusal_text_jsonl": args.refusal_text_jsonl,
            "refusal_vision_jsonl": args.refusal_vision_jsonl,
            "max_text_samples": args.max_text_samples,
            "max_vision_samples": args.max_vision_samples,
            "layers": layer_indices,
            "top_k_per_layer": args.top_k_per_layer,
            "pool_last_n_tokens": args.pool_last_n_tokens,
            "used_text_samples": used_text,
            "used_vision_samples": used_vision,
        },
    }

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[SUMMARY]")
    print(f"  manual_tokens:        {len(manual_tokens)}")
    print(f"  new_tokens:           {len(new_tokens_strs)}")
    print(f"  final RTS size:       {len(final_tokens)}")
    print(f"  used_text_samples:    {used_text}")
    print(f"  used_vision_samples:  {used_vision}")
    print(f"  out_json:             {args.out_json}")
    print(f"  out_csv_new:          {args.out_csv_new}")


if __name__ == "__main__":
    main()
