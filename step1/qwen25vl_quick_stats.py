#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen25vl_quick_stats.py

读取 work/qwen25vl/qwen25vl_step1_balanced_labels.jsonl，
统计 Attack / Refusal / Safe (Other) 的数量和比例，
以及 LG3 / GPT-4o 行为标签分布。
"""

import json
from pathlib import Path
from collections import Counter


def main():
    path = Path("work/qwen25vl/qwen25vl_step1_balanced_inference.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    c_group = Counter()
    c_lg3 = Counter()
    c_gpt = Counter()
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1
            g = obj.get("group", "Unknown")
            lg = obj.get("lg3_label", "Unknown")
            gp = obj.get("gpt4o_label", "Unknown")
            c_group[g] += 1
            c_lg3[lg] += 1
            c_gpt[gp] += 1

    print(f"Total samples: {total}\n")

    print("=== Group (A/R/S/Other) counts ===")
    for k, v in sorted(c_group.items()):
        print(f"{k}: {v}")
    print("\n=== Group rates ===")
    for k, v in sorted(c_group.items()):
        print(f"{k}: {v/total:.3f}")

    print("\n=== LG3 labels ===")
    for k, v in sorted(c_lg3.items()):
        print(f"{k}: {v}")

    print("\n=== GPT-4o behavior labels ===")
    for k, v in sorted(c_gpt.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
