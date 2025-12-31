#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen25vl_smoketest.py

用 Qwen/Qwen2.5-VL-7B-Instruct 在一条 VSB 视频上做一次简单推理，
确认环境和视频读入都正常。
"""

import argparse
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    ap.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="例如: data/vsb_videos/video/10_Hate/Religious_Intolerance/9.mp4",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="Describe what happens in the video.",
    )
    ap.add_argument(
        "--video-fps",
        type=float,
        default=1.0,
        help="采样帧率，默认为 1 FPS。",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    return ap.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"[INFO] Loading model: {args.model_id}")
    # 不显式开 flash_attn，避免你之前遇到的错误
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    print(f"[INFO] Model device: {model.device}")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": str(video_path)},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    print("[INFO] Building inputs...")
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        video_fps=args.video_fps,
    ).to(model.device)

    print("[INFO] Running generation...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    # 和官方示例一致的切片方式
    generated_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    outputs = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    answer = outputs[0].strip()
    print("\n[Qwen2.5-VL 输出]:")
    print(answer)


if __name__ == "__main__":
    main()
