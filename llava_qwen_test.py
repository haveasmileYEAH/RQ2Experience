import os
os.environ.setdefault("TRANSFORMERS_NO_FLASH_ATTENTION_2", "1")

import sys
from pathlib import Path
import torch

llava_path = "/home/RQ2Experience/model/LLaVA-NeXT"
if llava_path not in sys.path:
    sys.path.insert(0, llava_path)

from llava.model.builder import load_pretrained_model


def main():
    pretrained_id = "lmms-lab/LLaVA-Video-7B-Qwen2"

    # 关键修改：显式指定 attn_implementation="eager"
    tokenizer, model, image_processor, ctx_len = load_pretrained_model(
        pretrained_id,
        None,
        "llava_qwen",
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager"  # 强制使用传统的 Attention 实现，不检查 Flash-Attn
    )

    print("[OK] model loaded:", type(model))

if __name__ == "__main__":
    main()
