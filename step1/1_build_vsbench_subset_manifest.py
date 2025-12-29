#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm


def load_json_list(path: Path):
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    # 兼容两种常见格式：list 或 {"data":[...]}
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        obj = obj["data"]
    if not isinstance(obj, list):
        raise ValueError(f"Unexpected JSON format in {path}: expected list or {{'data': list}}")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", type=str, default="BAAI/Video-SafetyBench")
    ap.add_argument("--vsb_metadata_file", type=str, default="benign_data.json",
                    help="Use benign_data.json to get per-video benign prompt (question).")
    ap.add_argument("--raw_dir", type=str, default="data/vsb_meta",
                    help="Where to download metadata json via snapshot_download.")
    ap.add_argument("--n_per_category", type=int, default=30)
    ap.add_argument("--min_per_category", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_manifest", type=str, default="data/manifest_step1.jsonl")
    ap.add_argument("--out_stats", type=str, default="work/manifest_step1_stats.json")
    ap.add_argument("--force_redownload", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)

    root = Path(".").resolve()
    raw_dir = root / args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download ONLY metadata json (small). Dataset is gated: must accept terms + HF login.
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(raw_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[args.vsb_metadata_file],
        force_download=args.force_redownload,
    )

    meta_path = raw_dir / args.vsb_metadata_file
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found after download: {meta_path}")

    rows = load_json_list(meta_path)

    # Group by category
    by_cat = defaultdict(list)
    for r in rows:
        cat = r.get("category", "UNKNOWN")
        by_cat[cat].append(r)

    cats = sorted(by_cat.keys())
    if not cats:
        raise RuntimeError("No categories found. Check metadata schema.")

    # Sample per category
    selected = []
    insufficient = []
    for cat in cats:
        items = by_cat[cat]
        if len(items) < args.min_per_category:
            insufficient.append({"category": cat, "available": len(items)})
        k = min(args.n_per_category, len(items))

        rr = random.Random(args.seed + (abs(hash(cat)) % (10**9)))
        idx = list(range(len(items)))
        rr.shuffle(idx)
        pick = [items[i] for i in idx[:k]]
        selected.extend(pick)

    # Write manifest jsonl
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    with out_manifest.open("w", encoding="utf-8") as out:
        for r in tqdm(selected, desc="Writing manifest"):
            question_id = r.get("question_id", None)
            video_path = r.get("video_path", None) or r.get("video", None) or r.get("video_file", None)
            if not video_path:
                raise ValueError(f"Missing video_path in record keys={list(r.keys())}")

            # Use VSB benign question as the per-video prompt
            prompt = r.get("question", None)
            if not prompt:
                raise ValueError(f"Missing 'question' in benign metadata for question_id={question_id}")

            entry = {
                "uid": f"vsb_{question_id}" if question_id else f"vsb_{Path(video_path).stem}",
                "source": args.repo_id,

                "video_path": video_path,
                "category": r.get("category", "UNKNOWN"),
                "subcategory": r.get("subcategory", None),

                # 注意：这里的“benign”是指 query 表面无害，不代表视频语义无害
                "prompt_source": "vsb_benign_question",
                "prompt": prompt,

                # 额外元信息
                "vsb_question_id": question_id,
                "vsb_question_type": r.get("question_type", None),
                "vsb_harmful_intention": r.get("harmful_intention", None),
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Stats
    counts = Counter([x.get("category", "UNKNOWN") for x in selected])
    stats = {
        "repo_id": args.repo_id,
        "metadata_file": args.vsb_metadata_file,
        "seed": args.seed,
        "n_per_category_requested": args.n_per_category,
        "categories_found": len(cats),
        "total_selected": len(selected),
        "per_category_selected": dict(counts),
        "categories_with_low_availability": insufficient,
        "note": "Prompt is the original VSB benign question per video; this re-introduces prompt semantics (referential malice setting).",
    }
    out_stats = Path(args.out_stats)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote manifest: {out_manifest} (lines={len(selected)})")
    print(f"[OK] Wrote stats:    {out_stats}")


if __name__ == "__main__":
    main()
