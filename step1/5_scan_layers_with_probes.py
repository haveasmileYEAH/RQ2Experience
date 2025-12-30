#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step1 / 5_scan_layers_with_probes.py

功能：
- 对 step1_activations 中每一层的特征，训练简单 probe 做 R vs A 分类；
- 对每层输出 acc / auroc / f1（k-fold 平均），用来找“最安全敏感的层”。

输入：
- work/step1_activations/layer_{l}_features.npz
  - features: (N, D)
  - labels:   (N,)  0=R, 1=A

输出：
- work/step1_layer_probe_scores.csv
  每行：
    layer,probe_type,n_samples,n_splits,
    acc_mean,acc_std,auroc_mean,auroc_std,f1_mean,f1_std
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--activations_dir",
        type=str,
        default="work/step1_activations",
        help="存放 layer_{l}_features.npz 的目录",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="work/step1_layer_probe_scores.csv",
        help="输出的 layer-wise probe 结果表",
    )
    ap.add_argument(
        "--max_splits",
        type=int,
        default=5,
        help="StratifiedKFold 的最大折数，上限由少数类样本数决定",
    )
    ap.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="随机种子（打乱/划分用）",
    )
    return ap.parse_args()


def load_layer_files(activations_dir: Path) -> Dict[int, Path]:
    """
    搜索 activations_dir 下所有 layer_{l}_features.npz，返回 {layer_id: path}
    """
    layer_files: Dict[int, Path] = {}
    for p in activations_dir.glob("layer_*_features.npz"):
        name = p.stem  # e.g. "layer_8_features"
        parts = name.split("_")
        # 期望格式: layer_{l}_features
        if len(parts) >= 3 and parts[0] == "layer" and parts[2] == "features":
            try:
                lid = int(parts[1])
                layer_files[lid] = p
            except ValueError:
                continue
    return dict(sorted(layer_files.items(), key=lambda kv: kv[0]))


def decide_n_splits(y: np.ndarray, max_splits: int) -> int:
    """
    根据标签 y 的类别分布，自动选择 StratifiedKFold 的折数：
    - 不超过 max_splits
    - 不超过少数类样本数
    - 至少为 2
    """
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    n_splits = min(max_splits, int(min_count))
    if n_splits < 2:
        # 如果少数类太少，无法做 k-fold，只能退化为 2-fold（但这要求 min_count>=2）
        raise ValueError(f"Too few samples in minority class: {min_count}. Cannot do stratified K-fold.")
    return n_splits


def run_linear_probe_lr(
    X: np.ndarray,
    y: np.ndarray,
    max_splits: int,
    seed: int,
) -> Tuple[Dict[str, float], int]:
    """
    线性 Logistic Regression probe，带 StandardScaler + class_weight="balanced"。
    返回：
    - metrics: dict(acc_mean, acc_std, auroc_mean, auroc_std, f1_mean, f1_std)
    - n_splits 实际使用的折数
    """
    n_splits = decide_n_splits(y, max_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    accs, aucs, f1s = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Pipeline: StandardScaler + LogisticRegression
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                solver="liblinear",  # 小样本/高维友好
                max_iter=1000,
            ),
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # 对于 AUROC，需要连续得分：取正类概率
        # Pipeline 的最后一步是 LogisticRegression
        lr = clf.named_steps["logisticregression"]
        X_test_scaled = clf.named_steps["standardscaler"].transform(X_test)
        y_score = lr.predict_proba(X_test_scaled)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))

        # roc_auc_score 需要测试集中有两个类别
        if len(np.unique(y_test)) == 2:
            aucs.append(roc_auc_score(y_test, y_score))
        # 否则跳过该折 AUROC

        f1s.append(f1_score(y_test, y_pred, pos_label=1))

    def mean_std(values: List[float]) -> Tuple[float, float]:
        if len(values) == 0:
            return float("nan"), float("nan")
        return float(np.mean(values)), float(np.std(values))

    acc_mean, acc_std = mean_std(accs)
    auc_mean, auc_std = mean_std(aucs)
    f1_mean, f1_std = mean_std(f1s)

    metrics = {
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "auroc_mean": auc_mean,
        "auroc_std": auc_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
    }
    return metrics, n_splits


def run_mean_diff_probe(
    X: np.ndarray,
    y: np.ndarray,
    max_splits: int,
    seed: int,
) -> Tuple[Dict[str, float], int]:
    """
    Mean-Δμ 方向 probe：
    - 在训练折上计算 μ_R, μ_A；
    - w = μ_A - μ_R；
    - score = X_test @ w；
    - 用 score 做 roc_auc，按 1 为正类；
    - 分类阈值用 (proj_R + proj_A)/2 的中点。

    返回 metrics 同上。
    """
    n_splits = decide_n_splits(y, max_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    accs, aucs, f1s = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 计算训练集 μ_R, μ_A
        mask_R = (y_train == 0)
        mask_A = (y_train == 1)

        if mask_R.sum() == 0 or mask_A.sum() == 0:
            # 理论上 stratified 不会走到这里，但防御一下
            continue

        mu_R = X_train[mask_R].mean(axis=0)
        mu_A = X_train[mask_A].mean(axis=0)
        w = mu_A - mu_R  # 危险方向向量

        # 投影到一维
        proj_test = X_test @ w

        # 中点阈值（最近均值分类）
        proj_R = mu_R @ w
        proj_A = mu_A @ w
        thresh = 0.5 * (proj_R + proj_A)

        y_pred = (proj_test > thresh).astype(int)

        # AUROC：score 越大越偏向 A(1)
        if len(np.unique(y_test)) == 2:
            aucs.append(roc_auc_score(y_test, proj_test))

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, pos_label=1))

    def mean_std(values: List[float]) -> Tuple[float, float]:
        if len(values) == 0:
            return float("nan"), float("nan")
        return float(np.mean(values)), float(np.std(values))

    acc_mean, acc_std = mean_std(accs)
    auc_mean, auc_std = mean_std(aucs)
    f1_mean, f1_std = mean_std(f1s)

    metrics = {
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "auroc_mean": auc_mean,
        "auroc_std": auc_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
    }
    return metrics, n_splits


def main():
    args = parse_args()
    activations_dir = Path(args.activations_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    layer_files = load_layer_files(activations_dir)
    if not layer_files:
        raise RuntimeError(f"No layer_*_features.npz found in {activations_dir}")

    print(f"[INFO] Found {len(layer_files)} layers: {list(layer_files.keys())}")

    rows_out = []

    for layer_id, path in layer_files.items():
        data = np.load(path)
        X = data["features"].astype(np.float32)
        y = data["labels"].astype(int)

        n_samples, dim = X.shape
        print(f"[LAYER {layer_id}] X shape = {X.shape}, label distribution:", np.bincount(y))

        # Probe 1: Logistic Regression
        try:
            metrics_lr, n_splits_lr = run_linear_probe_lr(X, y, args.max_splits, args.random_seed)
            rows_out.append({
                "layer": layer_id,
                "probe_type": "linear_lr",
                "n_samples": n_samples,
                "n_splits": n_splits_lr,
                **metrics_lr,
            })
        except Exception as e:
            print(f"[WARN] linear_lr probe failed on layer {layer_id}: {e}")

        # Probe 2: Mean-Δμ direction
        try:
            metrics_md, n_splits_md = run_mean_diff_probe(X, y, args.max_splits, args.random_seed)
            rows_out.append({
                "layer": layer_id,
                "probe_type": "mean_diff",
                "n_samples": n_samples,
                "n_splits": n_splits_md,
                **metrics_md,
            })
        except Exception as e:
            print(f"[WARN] mean_diff probe failed on layer {layer_id}: {e}")

    # 写 CSV
    fieldnames = [
        "layer", "probe_type", "n_samples", "n_splits",
        "acc_mean", "acc_std",
        "auroc_mean", "auroc_std",
        "f1_mean", "f1_std",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows_out, key=lambda r: (r["probe_type"], r["layer"])):
            writer.writerow(row)

    print(f"[OK] Layer-wise probe scores written to: {out_csv}")


if __name__ == "__main__":
    main()
