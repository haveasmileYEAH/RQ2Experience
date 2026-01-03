import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ------------------------------------------------
# 兼容老 torch 的小补丁：修掉 is_autocast_enabled(device_type) 报错
# ------------------------------------------------
def _safe_patch_torch_autocast():
    import inspect

    try:
        sig = inspect.signature(torch.is_autocast_enabled)
        # 老版本没有参数，这时候我们包一层，忽略传入的 device_type
        if len(sig.parameters) == 0:
            orig = torch.is_autocast_enabled

            def _wrapped(*args, **kwargs):
                return orig()

            torch.is_autocast_enabled = _wrapped
    except Exception as e:
        print("[WARN] failed to patch torch.is_autocast_enabled:", repr(e))


_safe_patch_torch_autocast()

# 关掉 flash-attn，强制用 SDPA
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


# ------------------------------------------------
# 工具函数：均匀抽帧（decord）
# ------------------------------------------------
def sample_video_frames_uniform(video_path: str, num_frames: int) -> List[Image.Image]:
    import decord

    decord.bridge.set_bridge("native")

    vr = decord.VideoReader(video_path)
    n = len(vr)
    if n == 0:
        raise RuntimeError(f"Empty video: {video_path}")

    if num_frames <= 1:
        idxs = [n // 2]
    else:
        idxs = np.linspace(0, n - 1, num_frames).round().astype(int).tolist()

    frames: List[Image.Image] = []
    for i in idxs:
        arr = vr[i].asnumpy()
        frames.append(Image.fromarray(arr).convert("RGB"))
    return frames


# ------------------------------------------------
# 载入拒绝方向（多层）
# ------------------------------------------------
def load_refusal_directions(
    direction_dir: str, layer_ids: List[int]
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[str, Any]]:
    """
    direction_dir: work/refusal_direction
    layer_ids: 比如 [13, 27, 28]
    """
    ddir = Path(direction_dir)
    meta_path = ddir / "refusal_direction_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta file not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    directions: Dict[int, np.ndarray] = {}

    for lid in layer_ids:
        f = ddir / f"layer{lid:02d}.npy"
        if not f.exists():
            raise FileNotFoundError(f"direction file for layer {lid} not found: {f}")
        v = np.load(f)
        v = v.astype("float32")
        v_norm = np.linalg.norm(v)
        if v_norm > 0:
            v = v / v_norm
        directions[lid] = v

    lambda_weights: Dict[int, float] = {}
    if "lambda_scores" in meta and isinstance(meta["lambda_scores"], list):
        lam_list = meta["lambda_scores"]
        for lid in layer_ids:
            if 0 <= lid < len(lam_list):
                lambda_weights[lid] = float(lam_list[lid])

    return directions, lambda_weights, meta


# ------------------------------------------------
# 构造 Qwen2.5-VL 输入（多帧 + 文本）
# ------------------------------------------------
def build_qwen_inputs(processor, images: List[Image.Image], prompt: str, device: torch.device):
    contents = [{"type": "image", "image": img} for img in images]
    contents.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": contents}]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # --- 必须修改：加入像素限制，防止多帧视频导致视觉 Token 爆炸 ---
    inputs = processor(
        text=[chat_text],
        images=[images],
        return_tensors="pt",
        min_pixels=224 * 224,
        max_pixels=448 * 448, 
    )

    for k in list(inputs.keys()):
        if isinstance(inputs[k], torch.Tensor):
            inputs[k] = inputs[k].to(device)
    return inputs


# ------------------------------------------------
# 对单个样本计算 multi-layer refusal score
# ------------------------------------------------
def compute_multilayer_score(
    outputs,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    directions: Dict[int, np.ndarray],
    lambda_weights: Dict[int, float],
    layer_ids: List[int],
) -> Tuple[float, Dict[int, float]]:
    # Qwen2.5-VL 的 hidden_states 是一个元组
    hidden_states = outputs.hidden_states  
    num_available_layers = len(hidden_states)

    # 打印一次长度，方便调试（可选）
    # print(f"DEBUG: hidden_states len = {num_available_layers}")

    # 确定最后一个 token 的位置
    mask = attention_mask[0]
    non_pad = torch.nonzero(mask, as_tuple=False)
    if non_pad.numel() == 0:
        last_idx = input_ids.shape[1] - 1
    else:
        last_idx = int(non_pad[-1, 0])

    per_layer_scores: Dict[int, float] = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for lid in layer_ids:
        # 核心修复：确保索引不越界
        # 通常 hidden_states[0] 是 embedding，[1] 是第 0 层
        # 如果模型有 28 层，hidden_states 长度应该是 29
        idx = lid + 1
        if idx >= num_available_layers:
            # 如果越界了，尝试取最后一层作为 fallback，或者跳过
            idx = num_available_layers - 1
            
        # 提取 hidden state 并转为 float32 防止 Numpy 报错
        h_tensor = hidden_states[idx][0, last_idx, :]
        h = h_tensor.detach().cpu().float().numpy()
        
        h_norm = np.linalg.norm(h)
        if h_norm > 0:
            h = h / h_norm

        d = directions[lid]
        s = float(np.dot(h, d))
        per_layer_scores[lid] = s

        w = lambda_weights.get(lid, 1.0)
        if w > 0:
            weighted_sum += w * s
            weight_total += abs(w)

    if weight_total > 0:
        score = weighted_sum / weight_total
    else:
        score = float(sum(per_layer_scores.values()) / len(per_layer_scores)) if per_layer_scores else 0.0

    return score, per_layer_scores


# ------------------------------------------------
# 从 manifest 里尽量提取视频路径 & 文本提示
# ------------------------------------------------
def get_video_path(row: Dict[str, Any]) -> str:
    for key in ["video_path", "video", "path"]:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    raise ValueError("No video_path/video/path field found in row")


def get_text_prompt(row: Dict[str, Any]) -> str:
    # 尽可能兼容你之前生成的 manifest / step1 结果
    for key in [
        "attack_text",
        "question",
        "vsb_question_text",
        "vsb_full_prompt",
        "prompt",
        "query",
        "text",
    ]:
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # 实在没有就给个 fallback（但这种情况对我们没太大意义）
    return "Please answer the question about this video based on its content."


# ------------------------------------------------
# 主流程
# ------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True, help="e.g. data/manifest_step1.jsonl")
    ap.add_argument("--video_root", type=str, default=".", help="root dir that contains video/ ...")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--refusal_direction_dir", type=str, default="work/refusal_direction")
    ap.add_argument("--layers", type=str, default="13,27,28", help="comma separated layer ids, e.g. 13,27,28")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = use all")
    ap.add_argument("--out_jsonl", type=str, default="work/refusal_direction/video_multilayer_scores.jsonl")
    ap.add_argument("--out_csv", type=str, default="work/refusal_direction/video_multilayer_scores.csv")

    args = ap.parse_args()

    layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]
    print("[CONFIG] layers =", layer_ids)
    print("[CONFIG] manifest =", args.manifest)

    # 确保输出目录存在
    out_jsonl_path = Path(args.out_jsonl)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 载入拒绝方向
    directions, lambda_weights, meta = load_refusal_directions(args.refusal_direction_dir, layer_ids)
    print("[INFO] loaded refusal directions from", args.refusal_direction_dir)
    print("[INFO] lambda_weights:", {lid: round(lambda_weights.get(lid, 0.0), 3) for lid in layer_ids})

    # 载入 Qwen2.5-VL
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    print(f"[LOAD] loading Qwen2.5-VL model: {args.model_id} (dtype={args.torch_dtype})")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device
    print("[INFO] model device:", device)

    # 读 manifest
    manifest_path = Path(args.manifest)
    total_rows = 0
    rows: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            rows.append(json.loads(line))

    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    print(f"[INFO] loaded {len(rows)} rows from {manifest_path} (total={total_rows})")

    # 主循环
    jsonl_f = open(out_jsonl_path, "w", encoding="utf-8")

    scores_all: List[float] = []
    errors = 0

    for idx, row in enumerate(tqdm(rows, desc="Eval video with refusal direction")):
        rid = row.get("id", f"row_{idx}")
        try:
            rel_video_path = get_video_path(row)
            video_path = rel_video_path
            if not os.path.isabs(video_path):
                video_path = os.path.join(args.video_root, rel_video_path)

            prompt = get_text_prompt(row)

            # 均匀抽帧
            frames = sample_video_frames_uniform(video_path, args.num_frames)

            # Qwen2.5-VL 前向
            inputs = build_qwen_inputs(processor, frames, prompt, device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

            score, per_layer_scores = compute_multilayer_score(
                outputs,
                input_ids,
                attention_mask,
                directions,
                lambda_weights,
                layer_ids,
            )
            scores_all.append(score)

            out_obj = {
                "id": rid,
                "video_path": rel_video_path,
                "prompt": prompt,
                "score_multilayer": score,
                "per_layer_scores": per_layer_scores,
                "vsb_question_type": row.get("vsb_question_type"),
                "vsb_harmful_intention": row.get("vsb_harmful_intention"),
            }
            jsonl_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            jsonl_f.flush() # 实时保存，防止崩溃丢失

        except Exception as e:
            errors += 1
            jsonl_f.write(
                json.dumps(
                    {
                        "id": rid,
                        "error": repr(e),
                        "video_path": row.get("video_path"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        
        # --- 必须增加的显存清理逻辑 ---
        finally:
            if 'inputs' in locals(): del inputs
            if 'outputs' in locals(): del outputs
            if 'frames' in locals(): del frames
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            # ---------------------------

    jsonl_f.close()

    # 简单统计
    if scores_all:
        scores_np = np.array(scores_all, dtype="float32")
        print("\n[SUMMARY]")
        print("  samples_ok:", len(scores_all))
        print("  errors:    ", errors)
        print("  score mean:", float(scores_np.mean()))
        print("  score std: ", float(scores_np.std()))
        print("  score min: ", float(scores_np.min()))
        print("  score max: ", float(scores_np.max()))
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            print(f"  score p{int(q*100):02d}:", float(np.quantile(scores_np, q)))
    else:
        print("\n[SUMMARY] no valid scores, all samples failed or empty.")

    # 顺便写个 CSV 方便你之后画图
    try:
        import csv

        with open(out_csv_path, "w", encoding="utf-8", newline="") as cf:
            writer = csv.writer(cf)
            header = ["id", "video_path", "score_multilayer"] + [
                f"layer_{lid}" for lid in layer_ids
            ]
            writer.writerow(header)

            with open(out_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if "score_multilayer" not in r:
                        continue
                    row_out = [
                        r.get("id"),
                        r.get("video_path"),
                        r.get("score_multilayer"),
                    ]
                    pls = r.get("per_layer_scores") or {}
                    for lid in layer_ids:
                        row_out.append(pls.get(str(lid)) or pls.get(lid) or 0.0)
                    writer.writerow(row_out)

        print("[OK] CSV written to", out_csv_path)
    except Exception as e:
        print("[WARN] failed to write CSV:", repr(e))


if __name__ == "__main__":
    main()
