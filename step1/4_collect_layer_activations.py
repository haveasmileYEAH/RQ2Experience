#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step1 / 4_collect_layer_activations.py

功能：
- 在 Step1 的平衡子集上，收集 LLaVA-Video-7B-Qwen2 指定层（或全部层）的中间激活；
- 对每个样本、每个层做池化，得到一个定长向量，保存成 .npz，供后续 RQ2 实验使用。

关键特性：
- video_root + video_path 方式拼接视频路径（适配 data/video/... 结构）；
- 默认对「全部 transformer 层」做激活收集（--layers auto/all）；
- 显式使用 attn_implementation="sdpa"，不依赖 flash-attn；
- 输入张量统一 cast 到 model 实际 dtype，避免 dtype mismatch。
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from tqdm import tqdm
from decord import VideoReader, cpu

# LLaVA-NeXT / LLaVA-Video
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="data/manifest_step1_balanced.jsonl",
                    help="平衡后的 manifest（仅包含 group ∈ {A,R} 的子集）")
    ap.add_argument("--video_root", type=str, default="data",
                    help="解压后的视频根目录（其下应包含 video/...）")
    ap.add_argument("--out_dir", type=str, default="work/step1_activations")

    # LLaVA-Video 模型配置
    ap.add_argument("--pretrained", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2",
                    help="HF 模型名或本地权重路径")
    ap.add_argument("--model_name", type=str, default="llava_qwen",
                    help="llava.builder 里的 model_name，Qwen2 对应 llava_qwen")
    ap.add_argument("--conv_template", type=str, default="qwen_1_5")

    # 视频帧采样
    ap.add_argument("--max_frames", type=int, default=16,
                    help="对每个视频最多采样多少帧")
    ap.add_argument("--fps", type=int, default=1,
                    help="目标采样 FPS（基础步长）")
    ap.add_argument("--with_time_instruction", action="store_true",
                    help="是否在文本里加入时长/采样时间信息")

    # 设备与精度
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--dtype",
        type=str,
        default="float16",              # 默认用 float16，避免 bfloat16 兼容性坑
        choices=["bfloat16", "float16"]
    )

    # 层选择：auto/all = 自动用全部层；否则用逗号分隔的层号
    ap.add_argument(
        "--layers",
        type=str,
        default="auto",
        help="层选择：'auto'/'all' 表示 0..L-1 全部 transformer 层；或 '8,12,16' 显式指定"
    )

    # 运行控制
    ap.add_argument("--limit", type=int, default=-1,
                    help="仅处理前 N 条样本做 smoke test；-1 表示全量")
    ap.add_argument("--seed", type=int, default=0)

    return ap.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception as e:
                print(f"[ERROR] Failed to parse JSON at line {i}: {e}")
    return rows


def load_video_frames(video_path: Path, max_frames_num: int, fps: int = 1):
    """
    按给定 fps 从全视频均匀抽帧，如帧数超过 max_frames_num，则再做一次均匀下采样到 max_frames_num。
    返回：
    - frames: numpy(T,H,W,3)
    - frame_time_str: "t0,t1,..."
    - video_time: 总时长（秒）
    """
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = vr.get_avg_fps()
    video_time = total_frame_num / avg_fps

    stride = max(1, round(avg_fps / fps))
    frame_idx = list(range(0, total_frame_num, stride))

    if len(frame_idx) > max_frames_num:
        frame_idx = np.linspace(0, total_frame_num - 1, num=max_frames_num, dtype=int).tolist()

    frame_time = [i / avg_fps for i in frame_idx]
    frames = vr.get_batch(frame_idx).asnumpy()  # (T,H,W,3)

    frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])
    return frames, frame_time_str, video_time


def get_transformer_layers(model) -> List[torch.nn.Module]:
    """
    尝试在 LLaVA-Video-Qwen2 中找到 transformer 层列表。
    典型结构：model.model.layers (ModuleList)
    """
    base = model
    if hasattr(model, "model"):
        base = model.model

    if hasattr(base, "layers"):
        layers = base.layers
        if isinstance(layers, (list, torch.nn.ModuleList)):
            return list(layers)

    # 兜底搜索常见字段
    for attr in ["layers", "h", "blocks", "transformer"]:
        if hasattr(base, attr):
            layers = getattr(base, attr)
            if isinstance(layers, (list, torch.nn.ModuleList)):
                return list(layers)

    raise RuntimeError("Could not find transformer layers in model structure.")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest_path = Path(args.manifest)
    video_root = Path(args.video_root)
    out_dir = Path(args.out_dir)

    rows = read_jsonl(manifest_path)
    if args.limit > 0:
        rows = rows[:args.limit]

    # 只保留 group ∈ {A, R} 的样本
    rows = [r for r in rows if r.get("group") in ("A", "R")]
    print(f"[INFO] Processing {len(rows)} samples (group in A/R).")

    # 1) 加载模型（显式使用 SDPA，避免 flash-attn2 依赖）
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        None,
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if args.device == "cuda" else args.device,
        attn_implementation="sdpa",
    )
    model.eval()

    # 保险：把所有 config.attn_implementation 锁成 sdpa
    if hasattr(model, "config") and hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "sdpa"
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "attn_implementation"):
            m.config.attn_implementation = "sdpa"

    # 记录模型实际 dtype，后面把输入 cast 成这个类型
    model_dtype = next(model.parameters()).dtype
    print(f"[INFO] Model loaded. model_dtype = {model_dtype}")

    # 2) 找到所有 transformer 层
    layers = get_transformer_layers(model)
    print(f"[INFO] Transformer layers found: {len(layers)}")

    # 3) 解析要收集的层号
    layers_arg = args.layers.lower()
    if layers_arg in ("auto", "all"):
        layer_ids = list(range(len(layers)))  # 0..L-1
        print(f"[INFO] Using ALL layers: {layer_ids}")
    else:
        layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]
        print(f"[INFO] Using selected layers: {layer_ids}")

    # 4) 注册 forward hooks
    layer_outputs: Dict[int, torch.Tensor] = {}

    def make_hook(layer_id: int):
        def hook(module, inputs, output):
            # 兼容 (hidden_states, cache, ...) 结构
            hs = output[0] if isinstance(output, tuple) else output
            layer_outputs[layer_id] = hs.detach().cpu()
        return hook

    hooks = []
    for lid in layer_ids:
        hooks.append(layers[lid].register_forward_hook(make_hook(lid)))

    # 5) 为每个 layer 准备特征容器
    features_by_layer: Dict[int, List[np.ndarray]] = {lid: [] for lid in layer_ids}
    meta = {"labels": [], "uids": [], "groups": [], "categories": []}

    # 6) 遍历样本，前向 + 抽特征
    for r in tqdm(rows, desc="Collecting Activations"):
        uid = r.get("uid")
        group = r.get("group")
        category = r.get("category", "UNKNOWN")

        # R=0, A=1
        label = 0 if group == "R" else 1

        # 路径：video_root + video_path（例如 data + "video/xxx/xxx.mp4"）
        video_rel = r.get("video_path")
        if not video_rel:
            print(f"[WARN] Skip uid={uid}: missing video_path")
            continue
        abs_video_path = video_root / video_rel

        if not abs_video_path.exists():
            print(f"[WARN] Skip missing video: {abs_video_path}")
            continue

        prompt = r.get("prompt") or r.get("safe_prompt")
        if not prompt:
            print(f"[WARN] Skip uid={uid}: missing prompt/safe_prompt")
            continue

        try:
            # A) 读视频并抽帧
            frames_np, frame_time_str, video_time = load_video_frames(
                abs_video_path, max_frames_num=args.max_frames, fps=args.fps
            )
            pixel_values = image_processor.preprocess(frames_np, return_tensors="pt")["pixel_values"]
            pixel_values = pixel_values.to(args.device, dtype=model_dtype)

            # B) 构造对话 prompt
            conv = copy.deepcopy(conv_templates[args.conv_template])
            if args.with_time_instruction:
                time_inst = (
                    f"The video lasts for {video_time:.2f} seconds. "
                    f"{len(frames_np)} frames are sampled at: {frame_time_str}."
                )
                question = f"{DEFAULT_IMAGE_TOKEN}\n{time_inst}\n{prompt}"
            else:
                question = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"

            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            full_prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(args.device)

            # C) 清空缓存，前向一次
            layer_outputs.clear()
            with torch.no_grad():
                model(input_ids, images=[pixel_values], modalities=["video"])

            # D) 针对每个 hook 的层做池化（目前对所有 token 均值池化）
            for lid in layer_ids:
                if lid not in layer_outputs:
                    raise RuntimeError(f"Layer {lid} has no captured output for uid={uid}")
                hs = layer_outputs[lid]  # (1, seq, dim)
                pooled = hs.mean(dim=1).squeeze(0).numpy()  # (dim,)
                features_by_layer[lid].append(pooled)

            # 只有在前向 & 抽特征都成功的情况下，才记录 meta
            meta["labels"].append(label)
            meta["uids"].append(uid)
            meta["groups"].append(group)
            meta["categories"].append(category)

        except Exception as e:
            print(f"[ERROR] Failed uid {uid}: {e}")

    # 7) 取消 hooks 并保存结果
    for h in hooks:
        h.remove()
    out_dir.mkdir(parents=True, exist_ok=True)

    for lid in layer_ids:
        feats = features_by_layer[lid]
        if not feats:
            print(f"[WARN] No features collected for layer {lid}, skip saving.")
            continue
        np.savez_compressed(
            out_dir / f"layer_{lid}_features.npz",
            features=np.stack(feats),
            labels=np.array(meta["labels"]),
            uids=np.array(meta["uids"]),
            groups=np.array(meta["groups"]),
            categories=np.array(meta["categories"]),
            layer_id=lid,
        )
        print(f"[OK] Saved layer {lid} features to {out_dir / f'layer_{lid}_features.npz'}")

    print(f"[SUCCESS] Activations saved to {out_dir}")


if __name__ == "__main__":
    main()
