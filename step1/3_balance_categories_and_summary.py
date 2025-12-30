#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step1 / 3_balance_categories_and_summary.py

功能：
1）读取 Step1/2 的行为标注结果（包含 group ∈ {A, R, Other}）；
2）统计各类别在 A/R/Other 下的数量；
3）策略：
   - 保留所有 R（拒绝样本）作为全局“安全簇”；
   - 对 A（攻击成功样本）按类别做平衡抽样：
       * 只保留 A 数量 ≥ min_A_per_category 的类别；
       * 每类 A 最多抽 max_A_per_category 条；
   - 不将 Other 纳入主 manifest（可后续单独分析）；
4）输出：
   - data/manifest_step1_balanced.jsonl：仅包含 R 全部 + A 平衡后子集；
   - work/step1_category_stats.json：原始分布统计 + A 平衡后的分布；
   - work/step1_balancing_report.json：记录本次抽样配置与各类采样情况。

使用示例：

  conda activate vsb_step1_lg3
  cd ~/VSB_Step1

  python step1/3_balance_categories_and_summary.py \
    --in_file work/step1_behavior_labels_lg3_4o.jsonl \
    --out_stats work/step1_category_stats.json \
    --out_manifest data/manifest_step1_balanced.jsonl \
    --out_report work/step1_balancing_report.json \
    --min_A_per_category 8 \
    --max_A_per_category 16 \
    --seed 0
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", type=str, required=True,
                    help="Step1/2 行为标注结果 JSONL，例如 work/step1_behavior_labels_lg3_4o.jsonl")
    ap.add_argument("--out_stats", type=str, default="work/step1_category_stats.json",
                    help="输出的统计信息 JSON")
    ap.add_argument("--out_manifest", type=str, default="data/manifest_step1_balanced.jsonl",
                    help="输出的平衡版 manifest JSONL（仅包含 R + A_balanced）")
    ap.add_argument("--out_report", type=str, default="work/step1_balancing_report.json",
                    help="输出的 balancing 报告 JSON")
    ap.add_argument("--min_A_per_category", type=int, default=8,
                    help="某类 A 样本数若 < 该值，则视为样本不足，不纳入平衡子集")
    ap.add_argument("--max_A_per_category", type=int, default=16,
                    help="每类 A 样本最多抽取多少条，用于控制整体规模")
    ap.add_argument("--seed", type=int, default=0,
                    help="随机抽样的随机种子，便于复现实验")
    return ap.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def main():
    args = parse_args()
    random.seed(args.seed)

    in_path = Path(args.in_file)
    out_stats_path = Path(args.out_stats)
    out_manifest_path = Path(args.out_manifest)
    out_report_path = Path(args.out_report)

    assert in_path.exists(), f"in_file 不存在：{in_path}"

    rows = load_jsonl(in_path)

    # 1) 原始分布统计：overall + per-category
    overall_group_counter = Counter()
    per_category_group_counter = defaultdict(lambda: Counter())

    for r in rows:
        g = r.get("group")
        c = r.get("category", "UNKNOWN")
        overall_group_counter[g] += 1
        per_category_group_counter[c][g] += 1

    # 2) 拆出 R / A
    all_R_rows: List[Dict[str, Any]] = []
    A_by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        g = r.get("group")
        c = r.get("category", "UNKNOWN")

        if g == "R":
            all_R_rows.append(r)
        elif g == "A":
            A_by_category[c].append(r)
        else:
            # Other 先不纳入主 manifest，后续如有需要可单独处理
            continue

    # 3) 对 A 做类别过滤 + 抽样
    balanced_A_rows: List[Dict[str, Any]] = []
    balanced_A_counts_per_category: Dict[str, Counter] = defaultdict(Counter)

    kept_categories: List[str] = []
    dropped_categories: Dict[str, Dict[str, Any]] = {}

    for c, a_list in A_by_category.items():
        nA = len(a_list)
        if nA < args.min_A_per_category:
            # 样本不足的类别，记录在 dropped 列表中
            dropped_categories[c] = {
                "reason": "A_samples_too_few",
                "n_A": nA,
            }
            continue

        # 保留的类别
        kept_categories.append(c)
        n_pick = min(nA, args.max_A_per_category)

        # 随机抽样
        picked = random.sample(a_list, n_pick)
        for r in picked:
            balanced_A_rows.append(r)
            balanced_A_counts_per_category[c]["A"] += 1

    # 4) 构造最终 manifest：R 全部 + A 平衡子集
    final_rows: List[Dict[str, Any]] = []
    final_rows.extend(all_R_rows)
    final_rows.extend(balanced_A_rows)

    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest_path.open("w", encoding="utf-8") as f_out:
        for r in final_rows:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 5) 写出 stats：原始 + 平衡后
    stats = {
        "overall_group_counts": dict(overall_group_counter),
        "per_category_group_counts": {
            c: dict(cnt) for c, cnt in per_category_group_counter.items()
        },
        "balanced_A_counts_per_category": {
            c: dict(cnt) for c, cnt in balanced_A_counts_per_category.items()
        },
        "R_total_kept": len(all_R_rows),
        "config": {
            "min_A_per_category": args.min_A_per_category,
            "max_A_per_category": args.max_A_per_category,
            "seed": args.seed,
            "in_file": str(in_path),
            "out_manifest": str(out_manifest_path),
        },
        "balanced_total_samples": len(final_rows),
    }

    out_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with out_stats_path.open("w", encoding="utf-8") as f_stats:
        json.dump(stats, f_stats, ensure_ascii=False, indent=2)

    # 6) 写出 balancing 报告（更详细一点，方便写论文 / 复现）
    report = {
        "config": {
            "min_A_per_category": args.min_A_per_category,
            "max_A_per_category": args.max_A_per_category,
            "seed": args.seed,
        },
        "R": {
            "total_R": len(all_R_rows),
        },
        "A": {
            "total_A_before": sum(len(v) for v in A_by_category.values()),
            "total_A_after": len(balanced_A_rows),
            "kept_categories": kept_categories,
            "dropped_categories": dropped_categories,
            "balanced_A_counts_per_category": {
                c: dict(cnt) for c, cnt in balanced_A_counts_per_category.items()
            },
        },
        "paths": {
            "in_file": str(in_path),
            "out_manifest": str(out_manifest_path),
            "out_stats": str(out_stats_path),
        },
        "overall_group_counts": dict(overall_group_counter),
    }

    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    with out_report_path.open("w", encoding="utf-8") as f_report:
        json.dump(report, f_report, ensure_ascii=False, indent=2)

    print("[OK] Wrote balanced manifest to:", out_manifest_path)
    print("[OK] Wrote stats to:", out_stats_path)
    print("[OK] Wrote balancing report to:", out_report_path)
    print("[SUMMARY] balanced_total_samples =", len(final_rows))
    print("[SUMMARY] R_total =", len(all_R_rows),
          "| A_total_before =", report["A"]["total_A_before"],
          "| A_total_after =", len(balanced_A_rows))


if __name__ == "__main__":
    main()
