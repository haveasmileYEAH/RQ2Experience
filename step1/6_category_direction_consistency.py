#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step1 / 6_category_direction_consistency.py

功能：
1. 在指定层的 activations 上，按类别计算：
   - mu_R[c], mu_A[c], delta_L[c] = mu_R[c] - mu_A[c]
   - 以及全局 mu_R_global, mu_A_global, delta_L_global
2. 对每个层，输出：
   - layer_{l}_directions.npz: 保存上述向量
   - layer_{l}_cosine.csv: 各类别 delta_L 之间的余弦相似度矩阵
3. 可选：自动从 probe_scores_csv 中选出 AUROC 最高的层作为 L*，
   输出 step1_selected_safety_layer.json 方便 Step2 使用。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default="data/manifest_step1_balanced.jsonl",
        help="包含 uid / category / group 等信息的 manifest"
    )
    ap.add_argument(
        "--features_dir",
        type=str,
        default="work/step1_activations",
        help="step1_4 输出的 layer_*_features.npz 所在目录"
    )
    ap.add_argument(
        "--layers",
        type=str,
        default="21",
        help="要分析的层列表，如 '21' 或 '21,10,20'"
    )
    ap.add_argument(
        "--probe_scores_csv",
        type=str,
        default="work/step1_layer_probe_scores.csv",
        help="step1_5 的结果，用于自动选择最佳层（可选）"
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="work/step1_category_directions",
        help="输出目录"
    )
    ap.add_argument(
        "--min_per_group",
        type=int,
        default=1,
        help="每个类别内 R/A 至少各有多少样本才计算 delta_L"
    )
    ap.add_argument(
        "--primary_layer",
        type=int,
        default=-1,
        help="手动指定最终安全层 L*，若为 -1 则从 probe_scores_csv 中自动选"
    )
    ap.add_argument(
        "--probe_type_for_selection",
        type=str,
        default="linear_lr",
        help="自动选层时使用哪种 probe_type（默认 linear_lr）"
    )
    return ap.parse_args()


def read_manifest(path: Path) -> Dict[str, Dict]:
    """读取 manifest_step1_balanced.jsonl，为 uid -> row 构建索引（主要用于 sanity check）"""
    uid2row: Dict[str, Dict] = {}
    if not path.exists():
        print(f"[WARN] Manifest not found at {path}, will rely on npz metadata only.")
        return uid2row

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row.get("uid")
            if uid is not None:
                uid2row[uid] = row
    print(f"[INFO] Loaded manifest with {len(uid2row)} rows.")
    return uid2row


def compute_category_directions_for_layer(
    layer_id: int,
    features_path: Path,
    manifest_map: Dict[str, Dict],
    min_per_group: int = 1,
) -> Tuple[Dict, np.ndarray]:
    """
    对单个 layer 计算：
    - 全局 mu_R_global, mu_A_global, delta_L_global
    - 各类别 mu_R[c], mu_A[c], delta_L[c]
    返回：
      summary: dict（包含各种向量与统计）
      cos_matrix: np.ndarray 或 None（如果有效类别数 < 2 则为 None）
    """
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    data = np.load(features_path, allow_pickle=True)
    features = data["features"]           # (N, D)
    labels = data["labels"]              # (N,)
    uids = data["uids"]                  # (N,)
    groups = data["groups"]              # (N,)
    categories = data["categories"]      # (N,)

    N, D = features.shape
    print(f"[INFO][L{layer_id}] Loaded features: N={N}, D={D}")

    # 全局 R/A 掩码
    labels = labels.astype(int)
    mask_R = labels == 0
    mask_A = labels == 1

    if mask_R.sum() == 0 or mask_A.sum() == 0:
        print(f"[WARN][L{layer_id}] No R or A samples globally. R={mask_R.sum()}, A={mask_A.sum()}")
        return {}, None

    mu_R_global = features[mask_R].mean(axis=0)
    mu_A_global = features[mask_A].mean(axis=0)
    delta_global = mu_R_global - mu_A_global

    # 按类别统计
    cats_all = np.unique(categories)
    valid_cats: List[str] = []
    mu_R_list: List[np.ndarray] = []
    mu_A_list: List[np.ndarray] = []
    delta_list: List[np.ndarray] = []

    stats_per_cat = {}

    for cat in cats_all:
        mask_cat = categories == cat
        idx_cat = np.where(mask_cat)[0]
        n_cat = len(idx_cat)

        mask_R_cat = mask_cat & mask_R
        mask_A_cat = mask_cat & mask_A

        n_R_c = mask_R_cat.sum()
        n_A_c = mask_A_cat.sum()

        stats_per_cat[cat] = {
            "n_total": int(n_cat),
            "n_R": int(n_R_c),
            "n_A": int(n_A_c),
            "used": False
        }

        if n_R_c >= min_per_group and n_A_c >= min_per_group:
            mu_R_c = features[mask_R_cat].mean(axis=0)
            mu_A_c = features[mask_A_cat].mean(axis=0)
            delta_c = mu_R_c - mu_A_c

            valid_cats.append(cat)
            mu_R_list.append(mu_R_c)
            mu_A_list.append(mu_A_c)
            delta_list.append(delta_c)
            stats_per_cat[cat]["used"] = True
        else:
            print(f"[INFO][L{layer_id}] Skip category {cat}: n_R={n_R_c}, n_A={n_A_c} (min_per_group={min_per_group})")

    if not valid_cats:
        print(f"[WARN][L{layer_id}] No category has enough R/A samples, skip cosine matrix.")
        summary = {
            "layer": layer_id,
            "N": int(N),
            "D": int(D),
            "mu_R_global": mu_R_global,
            "mu_A_global": mu_A_global,
            "delta_global": delta_global,
            "categories_all": cats_all.tolist(),
            "stats_per_category": stats_per_cat,
            "valid_categories": [],
        }
        return summary, None

    mu_R_arr = np.stack(mu_R_list, axis=0)       # (C, D)
    mu_A_arr = np.stack(mu_A_list, axis=0)       # (C, D)
    delta_arr = np.stack(delta_list, axis=0)     # (C, D)

    # 计算余弦相似度矩阵
    # 先归一化
    def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norm + eps)

    delta_norm = l2_normalize(delta_arr)  # (C, D)
    cos_matrix = delta_norm @ delta_norm.T  # (C, C)

    summary = {
        "layer": layer_id,
        "N": int(N),
        "D": int(D),
        "mu_R_global": mu_R_global,
        "mu_A_global": mu_A_global,
        "delta_global": delta_global,
        "categories_all": cats_all.tolist(),
        "stats_per_category": stats_per_cat,
        "valid_categories": valid_cats,
        "mu_R_valid": mu_R_arr,
        "mu_A_valid": mu_A_arr,
        "delta_valid": delta_arr,
    }

    return summary, cos_matrix


def save_layer_outputs(
    layer_id: int,
    summary: Dict,
    cos_matrix: np.ndarray,
    out_dir: Path
):
    """保存该层的 npz 和 cosine 矩阵 csv。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"layer_{layer_id}"

    # 1) npz: 保存所有向量和元信息
    npz_path = out_dir / f"{base}_directions.npz"
    np.savez_compressed(
        npz_path,
        layer=summary["layer"],
        N=summary["N"],
        D=summary["D"],
        mu_R_global=summary["mu_R_global"],
        mu_A_global=summary["mu_A_global"],
        delta_global=summary["delta_global"],
        categories_all=np.array(summary["categories_all"], dtype=object),
        valid_categories=np.array(summary["valid_categories"], dtype=object),
        mu_R_valid=summary.get("mu_R_valid"),
        mu_A_valid=summary.get("mu_A_valid"),
        delta_valid=summary.get("delta_valid"),
        stats_per_category=json.dumps(summary["stats_per_category"], ensure_ascii=False),
    )
    print(f"[OK][L{layer_id}] Saved directions npz to {npz_path}")

    # 2) csv: 余弦相似度矩阵（仅对 valid_categories）
    if cos_matrix is not None and summary["valid_categories"]:
        cats = summary["valid_categories"]
        df = pd.DataFrame(cos_matrix, index=cats, columns=cats)
        csv_path = out_dir / f"{base}_cosine.csv"
        df.to_csv(csv_path, encoding="utf-8-sig")
        print(f"[OK][L{layer_id}] Saved cosine matrix csv to {csv_path}")
    else:
        print(f"[INFO][L{layer_id}] No valid categories for cosine matrix; skip csv.")


def auto_select_primary_layer(
    layers: List[int],
    probe_scores_csv: Path,
    probe_type: str = "linear_lr"
) -> int:
    """从 probe_scores_csv 中，按指定 probe_type 的 auroc_mean 在候选层里选出最优层。"""
    if not probe_scores_csv.exists():
        print(f"[WARN] probe_scores_csv not found at {probe_scores_csv}, cannot auto-select primary layer.")
        return -1

    df = pd.read_csv(probe_scores_csv)
    df = df[df["probe_type"] == probe_type]

    best_layer = -1
    best_auroc = -1.0

    for lid in layers:
        row = df[df["layer"] == lid]
        if row.empty:
            continue
        auroc = float(row["auroc_mean"].values[0])
        if auroc > best_auroc:
            best_auroc = auroc
            best_layer = lid

    if best_layer < 0:
        print("[WARN] None of the candidate layers found in probe_scores_csv; cannot auto-select.")
    else:
        print(f"[INFO] Auto-selected primary layer L*={best_layer} (probe_type={probe_type}, auroc_mean={best_auroc:.4f})")
    return best_layer


def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    probe_scores_csv = Path(args.probe_scores_csv)

    manifest_map = read_manifest(manifest_path)

    layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]
    layer_ids = sorted(set(layer_ids))

    all_layer_summaries = {}

    # 逐层计算
    for lid in layer_ids:
        feats_path = features_dir / f"layer_{lid}_features.npz"
        if not feats_path.exists():
            print(f"[WARN] Features for layer {lid} not found at {feats_path}, skip.")
            continue

        summary, cos_mat = compute_category_directions_for_layer(
            layer_id=lid,
            features_path=feats_path,
            manifest_map=manifest_map,
            min_per_group=args.min_per_group,
        )

        if not summary:
            continue

        save_layer_outputs(lid, summary, cos_mat, out_dir)
        all_layer_summaries[lid] = summary

    if not all_layer_summaries:
        print("[ERROR] No valid layer summaries produced, abort.")
        return

    # 选择最终 L*
    primary_layer = args.primary_layer
    if primary_layer < 0:
        primary_layer = auto_select_primary_layer(
            layers=list(all_layer_summaries.keys()),
            probe_scores_csv=probe_scores_csv,
            probe_type=args.probe_type_for_selection,
        )

    if primary_layer < 0 or primary_layer not in all_layer_summaries:
        print("[WARN] Primary layer not determined or not in computed layers; skip summary json.")
        return

    # 写 summary json
    summary_json = {
        "selected_layer": primary_layer,
        "layers_evaluated": sorted(list(all_layer_summaries.keys())),
        "selection_method": (
            "manual" if args.primary_layer >= 0 else
            f"auto_by_{args.probe_type_for_selection}_auroc"
        ),
        "delta_npz_path": str((out_dir / f"layer_{primary_layer}_directions.npz").as_posix()),
        "probe_scores_csv": str(probe_scores_csv.as_posix()),
    }

    summary_path = out_dir / "step1_selected_safety_layer.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote selected layer summary to: {summary_path}")
    print(f"[INFO] L* = {primary_layer}, delta_npz = {summary_json['delta_npz_path']}")


if __name__ == "__main__":
    main()
