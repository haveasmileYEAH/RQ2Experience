#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step2/1: Build refusal directions for a given layer.

Usage example:
    python step2/1_build_refusal_direction.py \
        --input-npz work/step1_activations/layer_9_features.npz \
        --layer-id 9 \
        --out-dir work/step2 \
        --flashattn spba

新增：
  - 支持命令行参数 --flashattn，用于统一控制注意力实现/FlashAttention2 开关。
  - 本脚本本身不加载 LLaVA/transformers，但会根据 --flashattn 设置环境变量
    TRANSFORMERS_NO_FLASH_ATTENTION_2，方便后续在同一进程内加载模型时继承。
"""

import os
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


EPS = 1e-9


# ========== FlashAttention / 注意力实现 控制 ==========

def resolve_flashattn_env(flashattn: str) -> str:
    """
    根据命令行参数 flashattn 设置环境变量 TRANSFORMERS_NO_FLASH_ATTENTION_2，
    并返回对应的注意力实现标识（供后续脚本复用）。

    参数 flashattn 允许：
      - "spba" / "sdpa" : 使用 PyTorch SDPA 实现，禁用 FlashAttention2
      - "eager"        : 使用 eager 实现，禁用 FlashAttention2
      - "fa2"          : 希望启用 FlashAttention2（本环境没装 flash_attn 的话会报错）
      - "none"         : 不做强制设置，保持 transformers 默认行为（谨慎使用）

    本脚本自身不会用到返回值 attn_impl，但会设置好环境变量，保持接口一致。
    """
    flashattn = flashattn.lower()
    if flashattn == "spba":
        attn_impl = "sdpa"
    elif flashattn == "sdpa":
        attn_impl = "sdpa"
    elif flashattn == "eager":
        attn_impl = "eager"
    elif flashattn == "fa2":
        attn_impl = "flash_attention_2"
    elif flashattn == "none":
        attn_impl = None
    else:
        # 理论上不会进来，因为 argparse 有 choices 限制
        attn_impl = None

    # 设置环境变量：只要不是明确要用 fa2，就禁止 FlashAttention2
    if attn_impl in ("sdpa", "eager", None):
        os.environ["TRANSFORMERS_NO_FLASH_ATTENTION_2"] = "1"
    else:
        # 想用 fa2 的情况：清除禁止标志（若之前被设置）
        os.environ.pop("TRANSFORMERS_NO_FLASH_ATTENTION_2", None)

    return attn_impl


# ========== 原有 Step2_1 逻辑 ==========

def _infer_layer_id_from_path(path: Path) -> Optional[int]:
    m = re.search(r"layer_(\d+)_", path.name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _ensure_str_array(arr: np.ndarray) -> np.ndarray:
    """
    Ensure an array of strings (possibly bytes) is converted to dtype '<U' (unicode).
    """
    if arr.dtype.kind in {"U", "O"}:
        return arr.astype(str)
    if arr.dtype.kind == "S":  # bytes
        return arr.astype(str)
    # Fallback: just cast to str
    return arr.astype(str)


def compute_refusal_directions(
    features: np.ndarray,
    groups: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute delta_L_simple, delta_L_symmetric, probe W, norm stats, and diagnostics.

    Args:
        features: (N, D) pooled hidden states.
        groups: (N,) array of string labels, at least containing "R" and "A".
        random_state: seed for random direction reproducibility.

    Returns:
        A dict with directions, stats, and intermediate values.
    """
    rng = np.random.RandomState(random_state)

    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    N, D = features.shape

    groups = _ensure_str_array(groups)
    mask_R = groups == "R"
    mask_A = groups == "A"

    n_R = int(mask_R.sum())
    n_A = int(mask_A.sum())
    n_all = int(N)

    if n_R == 0 or n_A == 0:
        raise ValueError(
            f"Need at least 1 R and 1 A sample to build directions, got n_R={n_R}, n_A={n_A}"
        )

    R_features = features[mask_R]  # (n_R, D)
    A_features = features[mask_A]  # (n_A, D)
    All_features = features  # (N, D)

    # Norm stats
    norms_all = np.linalg.norm(All_features, axis=1)
    L_avg_norm_all = float(np.mean(norms_all))

    norms_R = np.linalg.norm(R_features, axis=1)
    L_avg_norm_R = float(np.mean(norms_R))

    norms_A = np.linalg.norm(A_features, axis=1)
    L_avg_norm_A = float(np.mean(norms_A))

    # Unit-normalized R and A
    R_unit = R_features / (norms_R[:, None] + EPS)
    A_unit = A_features / (norms_A[:, None] + EPS)

    mu_R_unit = np.mean(R_unit, axis=0)  # (D,)
    mu_A_raw = np.mean(A_features, axis=0)  # (D,)
    mu_A_unit = np.mean(A_unit, axis=0)  # (D,)

    # DeltaL simple & symmetric
    delta_L_simple = mu_R_unit - mu_A_raw
    delta_L_symmetric = mu_R_unit - mu_A_unit

    def _unit(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < EPS:
            return v * 0.0
        return v / (norm + EPS)

    delta_L_simple_unit = _unit(delta_L_simple)
    delta_L_symmetric_unit = _unit(delta_L_symmetric)

    # Random direction for sanity check
    random_vec = rng.normal(size=(D,))
    random_unit_dir = _unit(random_vec)

    # Global mean for optional centering
    mu_all = np.mean(All_features, axis=0)  # (D,)

    # Train logistic regression probe R(1) vs A(0)
    X_probe = np.concatenate([R_features, A_features], axis=0)
    y_probe = np.concatenate([np.ones(n_R, dtype=int), np.zeros(n_A, dtype=int)], axis=0)

    # Center features by global mean to remove background bias
    X_probe_centered = X_probe - mu_all[None, :]

    clf = LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    clf.fit(X_probe_centered, y_probe)

    W_raw = clf.coef_.reshape(-1)  # (D,)
    W_unit = _unit(W_raw)

    # Training diagnostics
    y_pred = clf.predict(X_probe_centered)
    probe_train_acc = float(accuracy_score(y_probe, y_pred))

    try:
        y_prob = clf.predict_proba(X_probe_centered)[:, 1]
        probe_train_auc = float(roc_auc_score(y_probe, y_prob))
    except Exception:
        probe_train_auc = None

    # Alignment diagnostics
    def _cos(u: np.ndarray, v: np.ndarray) -> float:
        un = np.linalg.norm(u)
        vn = np.linalg.norm(v)
        if un < EPS or vn < EPS:
            return float("nan")
        return float(np.dot(u, v) / (un * vn + EPS))

    cos_deltaL_probeW_simple = _cos(delta_L_simple_unit, W_unit)
    cos_deltaL_probeW_symmetric = _cos(delta_L_symmetric_unit, W_unit)

    # Simple leave-one-out stability for delta_L_simple w.r.t R set
    deltaL_simple_R_loo_cos_mean = None
    deltaL_simple_R_loo_cos_min = None
    if n_R >= 3:
        loo_cos_list = []
        for i in range(n_R):
            R_loo = np.delete(R_features, i, axis=0)
            norms_R_loo = np.linalg.norm(R_loo, axis=1)
            R_loo_unit = R_loo / (norms_R_loo[:, None] + EPS)
            mu_R_loo_unit = np.mean(R_loo_unit, axis=0)
            deltaL_loo = mu_R_loo_unit - mu_A_raw
            deltaL_loo_unit = _unit(deltaL_loo)
            loo_cos = _cos(delta_L_simple_unit, deltaL_loo_unit)
            loo_cos_list.append(loo_cos)
        loo_cos_arr = np.array(loo_cos_list, dtype=float)
        deltaL_simple_R_loo_cos_mean = float(np.nanmean(loo_cos_arr))
        deltaL_simple_R_loo_cos_min = float(np.nanmin(loo_cos_arr))

    result: Dict[str, Any] = {
        "N": N,
        "D": D,
        "n_R": n_R,
        "n_A": n_A,
        "n_all": n_all,
        "L_avg_norm_all": L_avg_norm_all,
        "L_avg_norm_R": L_avg_norm_R,
        "L_avg_norm_A": L_avg_norm_A,
        "mu_R_unit": mu_R_unit,
        "mu_A_raw": mu_A_raw,
        "mu_A_unit": mu_A_unit,
        "delta_L_simple_unit": delta_L_simple_unit,
        "delta_L_symmetric_unit": delta_L_symmetric_unit,
        "deltaL_mode_default": "simple",
        "random_unit_dir": random_unit_dir,
        "probe_W_raw": W_raw,
        "probe_W_unit": W_unit,
        "mu_all": mu_all,
        "probe_train_acc": probe_train_acc,
        "probe_train_auc": probe_train_auc,
        "cos_sim_deltaL_probeW_simple": cos_deltaL_probeW_simple,
        "cos_sim_deltaL_probeW_symmetric": cos_deltaL_probeW_symmetric,
        "deltaL_simple_R_loo_cos_mean": deltaL_simple_R_loo_cos_mean,
        "deltaL_simple_R_loo_cos_min": deltaL_simple_R_loo_cos_min,
    }
    return result


def save_npz_and_summary(
    stats: Dict[str, Any],
    out_npz_path: Path,
    out_summary_path: Path,
    layer_id: Optional[int],
) -> None:
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare payload for npz (numpy arrays + scalars)
    npz_payload: Dict[str, Any] = {
        "delta_L_simple_unit": stats["delta_L_simple_unit"],
        "delta_L_symmetric_unit": stats["delta_L_symmetric_unit"],
        "deltaL_mode_default": np.array(stats["deltaL_mode_default"]),
        "probe_W_raw": stats["probe_W_raw"],
        "probe_W_unit": stats["probe_W_unit"],
        "random_unit_dir": stats["random_unit_dir"],
        "L_avg_norm_all": np.array(stats["L_avg_norm_all"], dtype=np.float32),
        "L_avg_norm_R": np.array(stats["L_avg_norm_R"], dtype=np.float32),
        "L_avg_norm_A": np.array(stats["L_avg_norm_A"], dtype=np.float32),
        "mu_R_unit": stats["mu_R_unit"],
        "mu_A_raw": stats["mu_A_raw"],
        "mu_A_unit": stats["mu_A_unit"],
        "mu_all": stats["mu_all"],
        "n_R": np.array(stats["n_R"], dtype=np.int64),
        "n_A": np.array(stats["n_A"], dtype=np.int64),
        "n_all": np.array(stats["n_all"], dtype=np.int64),
        "probe_train_acc": np.array(stats["probe_train_acc"], dtype=np.float32),
        "cos_sim_deltaL_probeW_simple": np.array(
            stats["cos_sim_deltaL_probeW_simple"], dtype=np.float32
        ),
        "cos_sim_deltaL_probeW_symmetric": np.array(
            stats["cos_sim_deltaL_probeW_symmetric"], dtype=np.float32
        ),
    }

    if stats.get("probe_train_auc") is not None:
        npz_payload["probe_train_auc"] = np.array(
            stats["probe_train_auc"], dtype=np.float32
        )
    if stats.get("deltaL_simple_R_loo_cos_mean") is not None:
        npz_payload["deltaL_simple_R_loo_cos_mean"] = np.array(
            stats["deltaL_simple_R_loo_cos_mean"], dtype=np.float32
        )
    if stats.get("deltaL_simple_R_loo_cos_min") is not None:
        npz_payload["deltaL_simple_R_loo_cos_min"] = np.array(
            stats["deltaL_simple_R_loo_cos_min"], dtype=np.float32
        )

    if layer_id is not None:
        npz_payload["layer_id"] = np.array(layer_id, dtype=np.int64)

    np.savez(out_npz_path, **npz_payload)

    # Prepare human-readable summary JSON
    summary: Dict[str, Any] = {
        "layer_id": int(layer_id) if layer_id is not None else None,
        "n_R": int(stats["n_R"]),
        "n_A": int(stats["n_A"]),
        "n_all": int(stats["n_all"]),
        "L_avg_norm_all": float(stats["L_avg_norm_all"]),
        "L_avg_norm_R": float(stats["L_avg_norm_R"]),
        "L_avg_norm_A": float(stats["L_avg_norm_A"]),
        "deltaL_mode_default": stats["deltaL_mode_default"],
        "probe_label_R": 1,
        "probe_label_A": 0,
        "probe_W_semantics": "W points to Refusal (y=1) region",
        "probe_train_acc": float(stats["probe_train_acc"]),
        "probe_train_auc": (
            float(stats["probe_train_auc"])
            if stats.get("probe_train_auc") is not None
            else None
        ),
        "cos_sim_deltaL_probeW_simple": float(
            stats["cos_sim_deltaL_probeW_simple"]
        ),
        "cos_sim_deltaL_probeW_symmetric": float(
            stats["cos_sim_deltaL_probeW_symmetric"]
        ),
        "deltaL_simple_R_loo_cos_mean": (
            float(stats["deltaL_simple_R_loo_cos_mean"])
            if stats.get("deltaL_simple_R_loo_cos_mean") is not None
            else None
        ),
        "deltaL_simple_R_loo_cos_min": (
            float(stats["deltaL_simple_R_loo_cos_min"])
            if stats.get("deltaL_simple_R_loo_cos_min") is not None
            else None
        ),
    }

    with out_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build refusal directions (delta_L, probe W) for a given layer from Step1 activations."
    )
    parser.add_argument(
        "--input-npz",
        type=str,
        required=True,
        help="Path to Step1 layer features NPZ (e.g., work/step1_activations/layer_9_features.npz).",
    )
    parser.add_argument(
        "--layer-id",
        type=int,
        default=-1,
        help="Layer id for bookkeeping. If -1, try to infer from filename pattern 'layer_{L}_features.npz'.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="work/step2",
        help="Output directory to store refusal_direction_layer_{L}.npz and summary JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (random direction, probe init).",
    )
    parser.add_argument(
        "--flashattn",
        type=str,
        default="spba",
        choices=["spba", "sdpa", "eager", "fa2", "none"],
        help=(
            "控制注意力实现 / FlashAttention2 行为，主要供后续加载 LLaVA 的脚本复用。\n"
            "  spba/sdpa : 使用 SDPA，并显式禁止 FlashAttention2 (默认推荐)\n"
            "  eager     : 使用 eager 注意力，并显式禁止 FlashAttention2\n"
            "  fa2       : 允许 FlashAttention2（本环境需安装 flash_attn，否则会报错）\n"
            "  none      : 不设置 FlashAttention 相关 env，保持 transformers 默认行为"
        ),
    )

    args = parser.parse_args()

    # 设置 FlashAttention 相关环境变量（本脚本本身不加载模型，主要为了统一接口）
    attn_impl = resolve_flashattn_env(args.flashattn)
    print(f"[INFO] flashattn arg = {args.flashattn}, mapped impl = {attn_impl}")
    print(
        f"[INFO] TRANSFORMERS_NO_FLASH_ATTENTION_2 = "
        f"{os.environ.get('TRANSFORMERS_NO_FLASH_ATTENTION_2', '<unset>')}"
    )

    input_npz_path = Path(args.input_npz)
    if not input_npz_path.is_file():
        raise FileNotFoundError(f"input NPZ not found: {input_npz_path}")

    if args.layer_id >= 0:
        layer_id: Optional[int] = int(args.layer_id)
    else:
        inferred = _infer_layer_id_from_path(input_npz_path)
        if inferred is None:
            print(
                "WARNING: Could not infer layer_id from filename; layer_id will be None in outputs."
            )
        layer_id = inferred

    print(f"[INFO] Loading features from: {input_npz_path}")
    data = np.load(input_npz_path, allow_pickle=True)
    if "features" not in data or "groups" not in data:
        raise KeyError(
            f"Expected 'features' and 'groups' in {input_npz_path}, got keys={list(data.keys())}"
        )

    features = np.asarray(data["features"])
    groups = np.asarray(data["groups"])

    print(f"[INFO] features shape: {features.shape}")
    print(f"[INFO] groups shape: {groups.shape}")

    stats = compute_refusal_directions(features, groups, random_state=args.seed)

    out_dir = Path(args.out_dir)
    if layer_id is None:
        out_npz = out_dir / "refusal_direction_layer_unknown.npz"
        out_json = out_dir / "refusal_direction_layer_unknown_summary.json"
    else:
        out_npz = out_dir / f"refusal_direction_layer_{layer_id}.npz"
        out_json = out_dir / f"refusal_direction_layer_{layer_id}_summary.json"

    print(f"[INFO] Saving directions to: {out_npz}")
    print(f"[INFO] Saving summary to: {out_json}")

    save_npz_and_summary(stats, out_npz, out_json, layer_id)

    print("[OK] Done. Key stats:")
    print(f"  n_R = {stats['n_R']}, n_A = {stats['n_A']}, n_all = {stats['n_all']}")
    print(
        f"  L_avg_norm_all = {stats['L_avg_norm_all']:.4f}, "
        f"L_avg_norm_R = {stats['L_avg_norm_R']:.4f}, "
        f"L_avg_norm_A = {stats['L_avg_norm_A']:.4f}"
    )
    print(
        f"  probe_train_acc = {stats['probe_train_acc']:.4f}, "
        f"probe_train_auc = {stats['probe_train_auc']}"
    )
    print(
        f"  cos(deltaL_simple, probeW) = {stats['cos_sim_deltaL_probeW_simple']:.4f}, "
        f"cos(deltaL_symmetric, probeW) = {stats['cos_sim_deltaL_probeW_symmetric']:.4f}"
    )
    if stats.get("deltaL_simple_R_loo_cos_mean") is not None:
        print(
            f"  deltaL_simple R-LOO cos mean = {stats['deltaL_simple_R_loo_cos_mean']:.4f}, "
            f"min = {stats['deltaL_simple_R_loo_cos_min']:.4f}"
        )


if __name__ == "__main__":
    main()
