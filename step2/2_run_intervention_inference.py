#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from tqdm import tqdm

# LLaVA / Qwen 相关
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.conversation import conv_templates

# 视频读取（按照官方示例，用 decord）
from decord import VideoReader, cpu


# -------------------------
# FlashAttention / SDPA 开关
# -------------------------
def resolve_flashattn(arg: str | None) -> str:
    """
    将命令行的 --flashattn 参数映射到 transformers 的 attn_implementation，
    同时设置环境变量禁止自动启用 FlashAttention2。
    """
    mapping = {
        None: "sdpa",
        "": "sdpa",
        "spba": "sdpa",              # 你习惯写 spba，这里映射到 sdpa
        "sdpa": "sdpa",
        "fa2": "flash_attention_2",  # 如果以后安装了 flash_attn，可以用这个
        "flash2": "flash_attention_2",
        "none": "eager",
        "eager": "eager",
    }
    arg_norm = (arg or "").lower()
    attn_impl = mapping.get(arg_norm, "sdpa")

    print(f"[INFO] flashattn arg = {arg_norm or '(default)'}, mapped impl = {attn_impl}")

    # 如果不是 flash_attention_2，就显式禁止 transformers 自动尝试 FA2
    if attn_impl != "flash_attention_2":
        os.environ["TRANSFORMERS_NO_FLASH_ATTENTION_2"] = "1"
    else:
        os.environ.pop("TRANSFORMERS_NO_FLASH_ATTENTION_2", None)

    print(f"[INFO] TRANSFORMERS_NO_FLASH_ATTENTION_2 = {os.environ.get('TRANSFORMERS_NO_FLASH_ATTENTION_2', '0')}")
    return attn_impl


# -------------------------
# 读取 Step2_1 生成的拒绝方向
# -------------------------
def load_refusal_direction(
    npz_path: str,
    direction_type: str,
) -> Tuple[np.ndarray, float, int, str | None]:
    """
    从 Step2_1 的 .npz 里读出单位拒绝方向向量等信息。
    返回:
        direction_vec_unit: np.ndarray, shape (D,)
        L_avg_norm_all: float
        layer_id_in_npz: int
        deltaL_mode: str | None
    """
    npz = np.load(npz_path)
    keys = npz.files
    print(f"[INFO] Loaded direction from: {npz_path}")
    print(f"[INFO]   keys = {keys}")

    if direction_type == "deltaL_simple":
        direction_vec_unit = npz["delta_L_simple_unit"]
        deltaL_mode = "simple"
    elif direction_type == "deltaL_symmetric":
        direction_vec_unit = npz["delta_L_symmetric_unit"]
        deltaL_mode = "symmetric"
    elif direction_type == "probeW":
        direction_vec_unit = npz["probe_W_unit"]
        deltaL_mode = None
    else:
        raise ValueError(f"Unknown direction_type: {direction_type}")

    L_avg_norm_all = float(npz["L_avg_norm_all"])
    layer_id_in_npz = int(npz["layer_id"])

    # 有些版本我在 Step2_1 里写的是 deltaL_mode_default，这里兼容一下
    if "deltaL_mode_default" in keys and deltaL_mode is None:
        deltaL_mode = str(npz["deltaL_mode_default"])

    print(f"[INFO]   L_avg_norm_all = {L_avg_norm_all:.4f}")
    print(f"[INFO]   layer_id_in_npz = {layer_id_in_npz}")
    print(f"[INFO]   direction_type = {direction_type}, deltaL_mode = {deltaL_mode}")
    return direction_vec_unit, L_avg_norm_all, layer_id_in_npz, deltaL_mode


# -------------------------
# manifest 读取（JSONL）
# -------------------------
def load_manifest(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    print(f"[INFO] Loaded {len(data)} samples from manifest: {path}")
    return data


# -------------------------
# 视频加载（完全照抄官方示例逻辑）
# -------------------------
def load_video_frames(
    video_path: str,
    max_frames_num: int,
    fps_div: int = 1,
    force_sample: bool = True,
) -> Tuple[np.ndarray, str, float]:
    """
    使用 decord 读取视频，并返回:
      - frames: np.ndarray, shape (T, H, W, 3)
      - frame_time_str: "0.00s,0.67s,..."
      - video_time: float, 秒
    逻辑与模型卡上的示例保持一致。
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = vr.get_avg_fps()
    video_time = float(total_frame_num) / float(avg_fps)

    # 抽帧策略：与 HF 示例类似
    fps = round(avg_fps / fps_div)
    frame_idx = [i for i in range(0, len(vr), max(fps, 1))]
    frame_time = [i / fps for i in frame_idx]

    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / avg_fps for i in frame_idx]

    frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])
    frames = vr.get_batch(frame_idx).asnumpy()  # (T, H, W, 3)

    return frames, frame_time_str, video_time


# -------------------------
# 找到 Qwen 的某一层并注册 hook
# -------------------------
def get_target_layer_module(model: torch.nn.Module, layer_id: int) -> torch.nn.Module:
    """
    针对 LlavaQwenForCausalLM，找到底层 Qwen2 的第 layer_id 层。
    日志上你已经看到：
        Model Class: LlavaQwenForCausalLM
        get_target_layer_module: using LlavaQwenModel.layers[10] (len=28)
    这里就是复用这一逻辑。
    """
    # LLaVA-NeXT 的 Qwen2 封装里一般有 get_model()
    base = None
    if hasattr(model, "get_model"):
        base = model.get_model()
    elif hasattr(model, "language_model"):
        base = model.language_model
    else:
        raise RuntimeError("Model has no attribute 'get_model' or 'language_model'; please adapt get_target_layer_module.")

    # Qwen2 decoder 列表一般在 .layers 里
    if hasattr(base, "layers"):
        layers = base.layers
    elif hasattr(base, "model") and hasattr(base.model, "layers"):
        layers = base.model.layers
    else:
        raise RuntimeError("Cannot find '.layers' in base language model; please inspect model architecture.")

    if not (0 <= layer_id < len(layers)):
        raise ValueError(f"layer_id={layer_id} out of range (len={len(layers)})")

    target = layers[layer_id]
    print(f"[INFO] get_target_layer_module: using {type(base).__name__}.layers[{layer_id}] (len={len(layers)})")
    print(f"[INFO] Hook registered on layer_id={layer_id}: {type(target).__name__}")
    return target


def make_refusal_hook(direction_vec_unit: np.ndarray, lambda_actual: float):
    """
    返回一个 forward_hook，用于在指定层对 hidden state 做加性偏移：
        h_new = h_old + lambda_actual * direction_vec_unit
    direction_vec_unit: 已经是单位范数方向，shape (D,)
    lambda_actual: 已缩放到与层范数同量级的标量
    """
    direction_np = direction_vec_unit.astype(np.float32)

    def hook_fn(module, inputs, output):
        # output 可能是 tensor 或 tuple；兼容两种情况
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        # hidden 形状：(B, T, D) 或 (T, D)
        # 将 direction_np 转成与 hidden 同 dtype / device，并通过 unsqueeze 做广播
        shift = torch.from_numpy(direction_np).to(device=hidden.device, dtype=hidden.dtype)
        while shift.dim() < hidden.dim():
            shift = shift.unsqueeze(0)  # -> (1, D) 或 (1, 1, D) 等

        hidden_new = hidden + lambda_actual * shift

        if rest is None:
            return hidden_new
        else:
            return (hidden_new, *rest)

    return hook_fn


# -------------------------
# 单条样本的多模态推理
# -------------------------
def run_model_on_sample(
    model,
    tokenizer,
    image_processor,
    sample: Dict[str, Any],
    device: torch.device,
    conv_template: str = "qwen_1_5",
    max_frames_num: int = 64,
    max_new_tokens: int = 512,
) -> str:
    """
    按照 LLaVA-Video-7B-Qwen2 模型卡的示例，对单条 (video + prompt) 做推理，
    唯一差别是：
      - video_path 从 sample["video_abs"] 或 sample["video_path"] 里拿；
      - prompt 使用 sample["prompt"]；
      - time_instruction 根据实际 frame_time / video_time 动态构建。
    hook 已经由外层注册，这里只管正常推理。
    """
    # 1. 找视频路径
    video_path = sample.get("video_abs") or sample.get("video_path")
    if video_path is None:
        raise ValueError(f"Sample missing 'video_abs' / 'video_path': {sample}")

    if not os.path.isabs(video_path):
        # 兼容相对路径（例如 data/vsb_videos/...），以 repo 根目录为基准
        video_path = str(Path(video_path).resolve())

    # 2. 加载视频帧（decord）
    frames, frame_time_str, video_time = load_video_frames(
        video_path,
        max_frames_num=max_frames_num,
        fps_div=1,
        force_sample=True,
    )

    # 3. 用 image_processor.preprocess，把 (T, H, W, 3) -> (T, C, H, W) 的 pixel_values
    processed = image_processor.preprocess(frames, return_tensors="pt")
    pixel_values = processed["pixel_values"]  # (T, C, H, W)

    # 对齐模型 dtype / device
    param = next(model.parameters())
    param_dtype = param.dtype
    video_tensor = pixel_values.to(device=device, dtype=param_dtype)

    # 模型期望 images 是一个 list[Tensor]，每个元素 shape: (T, C, H, W)
    images = [video_tensor]

    # 4. 构造 time_instruction + question
    num_frames = video_tensor.shape[0]
    time_instruction = (
        f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are "
        f"uniformly sampled from it. These frames are located at {frame_time_str}."
    )

    user_prompt = sample.get("prompt") or ""
    question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{user_prompt}"

    # 5. conv_template 组 chat prompt
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    # 6. tokenizer_image_token 插入 IMAGE_TOKEN_INDEX
    input_ids = tokenizer_image_token(
        prompt_question,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    # 7. 调用 model.generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            images=images,               # 关键：用 images=video，而不是 videos=
            modalities=["video"],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )

    # 8. 解码
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return text


# -------------------------
# 主入口
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # 样本 / 方向 / 输出设置
    parser.add_argument("--manifest", type=str, required=True,
                        help="JSONL manifest 文件，例如 mismatched_cases.jsonl")
    parser.add_argument("--direction-npz", type=str, required=True,
                        help="Step2_1 输出的 refusal_direction_layer_{L}.npz")
    parser.add_argument("--layer-id", type=int, required=True,
                        help="目标 hook 的层号（与 Step1 保持一致的 0-based 编号）")
    parser.add_argument("--direction-type", type=str, default="probeW",
                        choices=["deltaL_simple", "deltaL_symmetric", "probeW"])
    parser.add_argument("--lambda-scales", type=float, nargs="+", required=True,
                        help="例如: --lambda-scales -0.1 -0.05 0.0 0.05 0.1")
    parser.add_argument("--out-dir", type=str, default="work/step2",
                        help="输出目录")

    # 模型与推理设置
    parser.add_argument("--model-path", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--model-name", type=str, default="llava_qwen")
    parser.add_argument("--flashattn", type=str, default="spba",
                        help="fa2 / spba / sdpa / none 等，建议继续用 spba")
    parser.add_argument("--max-frames-num", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--conv-template", type=str, default="qwen_1_5")

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) FlashAttention / SDPA 配置
    attn_impl = resolve_flashattn(args.flashattn)

    # 2) 读取拒绝方向（Step2_1 输出）
    direction_vec_unit, L_avg_norm_all, layer_id_in_npz, deltaL_mode = load_refusal_direction(
        args.direction_npz,
        args.direction_type,
    )

    # 可选：检查 layer_id 是否一致（这里只是 warn，不强制）
    if layer_id_in_npz != args.layer_id:
        print(f"[WARN] layer_id in npz = {layer_id_in_npz}, but args.layer_id = {args.layer_id}. "
              f"确保你知道自己在干什么。")

    # 3) 读取样本 manifest
    samples = load_manifest(args.manifest)

    # 4) 加载 LLaVA-Video-7B-Qwen2 模型
    print(f"[INFO] Loading LLaVA model: {args.model_path}")
    tokenizer, model, image_processor, ctx_len = load_pretrained_model(
        args.model_path,
        None,
        args.model_name,
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()

    print(f"Model Class: {type(model).__name__}")
    print(f"[INFO] Model loaded. ctx_len = {ctx_len}")
    # 主设备（仅用于日志）
    main_device = next(model.parameters()).device
    print(f"[INFO] Model main device: {main_device}")

    # 5) 找到要 hook 的层
    target_module = get_target_layer_module(model, args.layer_id)

    # 6) 逐个 lambda_scale 运行推理
    for lambda_scale in args.lambda_scales:
        lambda_actual = float(lambda_scale) * float(L_avg_norm_all)
        print(f"[INFO] ===== lambda_scale={lambda_scale:+.4f}, lambda_actual={lambda_actual:+.4f} =====")

        # 为当前 lambda_scale 创建 hook
        hook_fn = make_refusal_hook(direction_vec_unit, lambda_actual)
        handle = target_module.register_forward_hook(hook_fn)

        # 输出文件命名
        lambda_tag = f"{lambda_scale:+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")
        out_path = out_dir / f"intervention_generations_layer{args.layer_id}_{args.direction_type}_lam{lambda_tag}.jsonl"
        print(f"[INFO] Writing generations to: {out_path}")

        with open(out_path, "w", encoding="utf-8") as fout, torch.no_grad():
            for sample in tqdm(samples, desc=f"lambda={lambda_scale:+.4f}", ncols=80):
                uid = sample.get("uid")
                category = sample.get("category")

                try:
                    answer_text = run_model_on_sample(
                        model=model,
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        sample=sample,
                        device=main_device,
                        conv_template=args.conv_template,
                        max_frames_num=args.max_frames_num,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as e:
                    print(f"[WARN] uid={uid} failed with error: {e}")
                    answer_text = ""
                    error_str = str(e)
                else:
                    error_str = None

                record = {
                    "uid": uid,
                    "category": category,
                    "layer_id": args.layer_id,
                    "direction_type": args.direction_type,
                    "deltaL_mode": deltaL_mode,
                    "lambda_scale": lambda_scale,
                    "lambda_actual": lambda_actual,
                    "model_path": args.model_path,
                    "flashattn_impl": attn_impl,
                    "prompt": sample.get("prompt"),
                    "video_path": sample.get("video_path"),
                    "video_abs": sample.get("video_abs"),
                    "answer_text": answer_text,
                    "error": error_str,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 当前 lambda_scale 完成后移除 hook
        handle.remove()
        print(f"[INFO] Done for lambda_scale={lambda_scale:+.4f}, results -> {out_path}")

    print("[OK] All lambda_scales finished.")


if __name__ == "__main__":
    main()
