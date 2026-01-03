#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_response_text(row: Dict[str, Any]) -> str:
    """尽量兼容不同 key 命名."""
    for k in ["model_response", "response", "output", "answer"]:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def is_refusal_row(row: Dict[str, Any]) -> bool:
    if "is_refusal" in row:
        return bool(row.get("is_refusal"))
    if "is_refusal_rule" in row:
        return bool(row.get("is_refusal_rule"))
    return False


def is_error_row(row: Dict[str, Any]) -> bool:
    err = row.get("error")
    return bool(err)


def normalize_token_str(tok: str) -> str:
    """
    用于过滤无意义 token：
    - 去掉常见前缀（如 SentencePiece 的 '▁'）
    - 去掉首尾空白
    - 如果全部是标点或空，则视为无效
    """
    if tok is None:
        return ""
    s = tok.replace("▁", " ").strip()
    if not s:
        return ""
    # 如果全是标点符号，也视为无效
    if all(ch in string.punctuation for ch in s):
        return ""
    return s


def build_token_stats(
    input_path: str,
    tokenizer_name: str,
    min_freq: int,
    min_responses: int,
    top_k: int,
    refusal_only: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    返回：候选 token 列表 + 一个 summary dict 用于打印。
    """
    tok = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        use_fast=False,
    )

    token_stats = {}  # token_id -> info dict
    total_rows = 0
    total_refusal_rows = 0
    total_error_rows = 0

    pbar = tqdm(
        read_jsonl(input_path),
        desc=f"Build RTS from {Path(input_path).name}",
        unit="resp",
    )

    for row in pbar:
        total_rows += 1
        if is_error_row(row):
            total_error_rows += 1
            continue

        # 只统计拒绝样本的话：
        ref = is_refusal_row(row)
        if refusal_only and not ref:
            continue
        if ref:
            total_refusal_rows += 1

        text = get_response_text(row)
        if not text:
            continue

        # tokenizer 编码
        encoded = tok(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = encoded["input_ids"]
        toks = tok.convert_ids_to_tokens(ids)

        seen_ids_this_resp = set()

        for tid, tstr in zip(ids, toks):
            norm = normalize_token_str(tstr)
            if not norm:
                continue

            # 初次见到该 token_id
            if tid not in token_stats:
                token_stats[tid] = {
                    "token_id": tid,
                    "token": norm,
                    "raw_token": tstr,
                    "freq_total": 0,
                    "num_responses": 0,
                    "example_response_snippet": "",
                }

            ts = token_stats[tid]
            ts["freq_total"] += 1

            if tid not in seen_ids_this_resp:
                ts["num_responses"] += 1
                seen_ids_this_resp.add(tid)
                # 保存一个示例片段（只记录第一条）
                if not ts["example_response_snippet"]:
                    ts["example_response_snippet"] = text[:200]

        pbar.set_postfix(
            rows=total_rows,
            refusals=total_refusal_rows,
            uniq_tokens=len(token_stats),
        )

    # 过滤 & 排序
    candidates: List[Dict[str, Any]] = []
    for ts in token_stats.values():
        if min_freq > 0 and ts["freq_total"] < min_freq:
            continue
        if min_responses > 0 and ts["num_responses"] < min_responses:
            continue
        candidates.append(ts)

    candidates.sort(key=lambda x: (x["num_responses"], x["freq_total"]), reverse=True)
    if top_k > 0 and len(candidates) > top_k:
        candidates = candidates[:top_k]

    summary = {
        "total_rows": total_rows,
        "total_refusal_rows": total_refusal_rows,
        "total_error_rows": total_error_rows,
        "uniq_tokens_raw": len(token_stats),
        "uniq_tokens_after_filter": len(candidates),
    }
    return candidates, summary


def write_csv(cands: List[Dict[str, Any]], out_path: str):
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "token",
        "token_id",
        "raw_token",
        "freq_total",
        "num_responses",
        "example_response_snippet",
    ]
    with out_p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ts in cands:
            row = {k: ts.get(k, "") for k in fieldnames}
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True,
                    help="qwen_refusals_*.jsonl")
    ap.add_argument("--model_id", type=str, required=True,
                    help="e.g. Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out_csv", type=str, required=True,
                    help="where to write candidate_tokens_*.csv")
    ap.add_argument("--min_freq", type=int, default=3,
                    help="minimum total frequency")
    ap.add_argument("--min_responses", type=int, default=3,
                    help="minimum number of responses containing this token")
    ap.add_argument("--top_k", type=int, default=300,
                    help="max number of tokens to output (0 = no limit)")
    ap.add_argument("--include_non_refusal", action="store_true",
                    help="if set, include non-refusal responses as well.")

    args = ap.parse_args()

    cands, summary = build_token_stats(
        input_path=args.input_jsonl,
        tokenizer_name=args.model_id,
        min_freq=args.min_freq,
        min_responses=args.min_responses,
        top_k=args.top_k,
        refusal_only=(not args.include_non_refusal),
    )
    write_csv(cands, args.out_csv)

    print("\n[SUMMARY]")
    print(f"  input_jsonl:              {args.input_jsonl}")
    print(f"  total_rows:               {summary['total_rows']}")
    print(f"  total_refusal_rows:       {summary['total_refusal_rows']}")
    print(f"  total_error_rows:         {summary['total_error_rows']}")
    print(f"  uniq_tokens_raw:          {summary['uniq_tokens_raw']}")
    print(f"  uniq_tokens_after_filter: {summary['uniq_tokens_after_filter']}")
    print(f"  out_csv:                  {args.out_csv}")


if __name__ == "__main__":
    main()
