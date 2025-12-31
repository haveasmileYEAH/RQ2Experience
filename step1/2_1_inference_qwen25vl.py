#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1 / 2_1_inference_qwen25vl.py

使用 Qwen/Qwen2.5-VL-7B-Instruct
对 manifest_step1_balanced.jsonl 中的样本跑视频推理，
输出 JSONL，可直接喂给 2_2_lg3_score.py 打标签。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen2.5-VL 模型 ID",
    )
    ap.add_argument(
        "--manifest",
        type=str,
        default="data/manifest_step1_balanced.jsonl",
        help="输入 manifest（与 Step1 相同结构）。",
    )
    ap.add_argument(
        "--out-file",
        type=str,
        default="work/qwen25vl/qwen25vl_step1_balanced_inference.jsonl",
        help="输出 JSONL 路径。",
    )
    ap.add_argument(
        "--video-root",
        type=str,
        default="",
        help="可选：如果 manifest 里 video_path 是相对路径，可以指定一个根目录。",
    )
    ap.add_argument(
        "--video-fps",
        type=float,
        default=1.0,
        help="Qwen2.5-VL 读取视频时的采样 FPS。",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="生成最大 token 数。",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="调试用，只跑前 N 条（-1 = 全部）。",
    )
    return ap.parse_args()


def resolve_video_path(sample: Dict[str, Any], video_root: str) -> str:
    """
    优先使用 manifest 中的 video_abs，其次 video_path。
    如果是相对路径且指定了 video_root，则拼接。
    """
    vp = sample.get("video_abs") or sample.get("video_path")
    if vp is None:
        raise ValueError(f"No video path in sample: {sample.get('uid')}")
    p = Path(vp)
    if not p.is_absolute():
        if video_root:
            p = Path(video_root) / p
        else:
            p = p  # 相对路径，默认相对仓库根目录
    return str(p)


def load_model_and_processor(model_id: str):
    print(f"[INFO] Loading Qwen2.5-VL model: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"[INFO] Model device: {model.device}")
    return model, processor


@torch.no_grad()
def generate_one(
    model,
    processor,
    video_path: str,
    prompt: str,
    video_fps: float = 1.0,
    max_new_tokens: int = 256,
) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        video_fps=video_fps,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    # 官方推荐的切片方式
    generated_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    outputs = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return outputs[0].strip()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"[INFO] Loading manifest: {manifest_path}")
    samples = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    if args.limit > 0:
        samples = samples[: args.limit]
        print(f"[INFO] Limit = {args.limit}, using first {len(samples)} samples")
    else:
        print(f"[INFO] Total samples = {len(samples)}")

    model, processor = load_model_and_processor(args.model_id)

    with out_path.open("w", encoding="utf-8") as fout:
        for sample in tqdm(samples, desc="Qwen2.5-VL inference"):
            uid = sample.get("uid")
            prompt = sample.get("prompt")
            if not prompt:
                raise ValueError(f"Sample {uid} has no 'prompt' field.")

            try:
                video_path = resolve_video_path(sample, args.video_root)
                if not os.path.exists(video_path):
                    raise FileNotFoundError(video_path)

                answer = generate_one(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    prompt=prompt,
                    video_fps=args.video_fps,
                    max_new_tokens=args.max_new_tokens,
                )
                sample["model_answer"] = answer
                sample["error"] = None
                sample["model_name"] = args.model_id
            except Exception as e:
                sample["model_answer"] = None
                sample["error"] = str(e)
                sample["model_name"] = args.model_id
                print(f"[WARN] uid={uid} failed: {e}")

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[OK] Qwen2.5-VL inference saved to: {out_path}")


if __name__ == "__main__":
    main()
