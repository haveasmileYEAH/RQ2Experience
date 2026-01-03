import argparse
import json
import csv
from pathlib import Path
from typing import Set


def load_tokens_from_json(path: Path) -> Set[str]:
    """Robustly load a set of tokens from a json / json-like structure."""
    data = json.loads(path.read_text(encoding="utf-8"))

    # case 1: dict with "tokens" (常见结构)
    if isinstance(data, dict):
        if isinstance(data.get("tokens"), list):
            return {str(t) for t in data["tokens"]}

        # rts_expanded_clean.json 结构: { tokens, manual_tokens, new_tokens_clean }
        tokens = set()
        for key, val in data.items():
            if isinstance(val, list):
                tokens.update(str(t) for t in val)
        return tokens

    # case 2: list 本身就是 tokens
    if isinstance(data, list):
        return {str(t) for t in data}

    return set()


def load_tokens_from_csv(path: Path) -> Set[str]:
    """从候选 token 统计 csv 中抽取 token 列"""
    tokens = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "token" not in (reader.fieldnames or []):
            print(f"[WARN] {path} 中没有 'token' 列，跳过该文件。")
            return tokens
        for row in reader:
            tok = (row.get("token") or "").strip()
            if tok:
                tokens.add(tok)
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rts_manual_json",
        type=str,
        required=True,
        help="例如 work/rts/rts_final_manual.json"
    )
    ap.add_argument(
        "--rts_expanded_json",
        type=str,
        required=True,
        help="例如 work/rts/rts_expanded_clean.json"
    )
    ap.add_argument(
        "--candidate_text_csv",
        type=str,
        default="",
        help="可选：文本攻击集的 token 统计表 (candidate_tokens_text.csv)"
    )
    ap.add_argument(
        "--candidate_vision_csv",
        type=str,
        default="",
        help="可选：图文攻击集的 token 统计表 (candidate_tokens_vision.csv)"
    )
    ap.add_argument(
        "--out_md",
        type=str,
        default="work/rts/rts_comparison_report.md",
        help="输出的 Markdown 报告路径"
    )
    ap.add_argument(
        "--out_final_json",
        type=str,
        default="work/rts/rts_final.json",
        help="最终 RTS 输出路径（一般直接采用 expanded_clean 的 tokens）"
    )

    args = ap.parse_args()

    manual_path = Path(args.rts_manual_json)
    expanded_path = Path(args.rts_expanded_json)
    out_md_path = Path(args.out_md)
    out_final_path = Path(args.out_final_json)

    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_final_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 加载手工 RTS + 扩展 RTS
    manual_tokens = load_tokens_from_json(manual_path)
    expanded_tokens = load_tokens_from_json(expanded_path)

    inter_manual_expanded = manual_tokens & expanded_tokens
    manual_only = manual_tokens - expanded_tokens
    expanded_only = expanded_tokens - manual_tokens

    print("=== RTS (manual vs expanded) ===")
    print("manual size:", len(manual_tokens))
    print("expanded size:", len(expanded_tokens))
    print("intersection:", len(inter_manual_expanded))
    print("manual-only:", len(manual_only))
    print("expanded-only:", len(expanded_only))

    # 2. 如有 text / vision 的候选 CSV，则做文本 vs 图文比较
    text_tokens = set()
    vision_tokens = set()

    if args.candidate_text_csv:
        p = Path(args.candidate_text_csv)
        if p.exists():
            text_tokens = load_tokens_from_csv(p)
            print(f"[INFO] loaded {len(text_tokens)} text-candidate tokens from {p}")
        else:
            print(f"[WARN] candidate_text_csv not found: {p}")

    if args.candidate_vision_csv:
        p = Path(args.candidate_vision_csv)
        if p.exists():
            vision_tokens = load_tokens_from_csv(p)
            print(f"[INFO] loaded {len(vision_tokens)} vision-candidate tokens from {p}")
        else:
            print(f"[WARN] candidate_vision_csv not found: {p}")

    inter_tv = text_tokens & vision_tokens
    text_only = text_tokens - vision_tokens
    vision_only = vision_tokens - text_tokens

    # 3. 生成最终 RTS（策略：使用 expanded_tokens，可换成 manual_tokens 或并集）
    final_tokens = sorted(expanded_tokens)

    out_final = {
        "tokens": final_tokens,
        "meta": {
            "source_manual": str(manual_path),
            "source_expanded": str(expanded_path),
            "note": (
                "final RTS 基于清洗后的 expanded RTS。"
                "manual RTS 用于保证核心拒绝词覆盖。"
            ),
        },
    }
    out_final_path.write_text(
        json.dumps(out_final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] wrote final RTS to: {out_final_path} (size={len(final_tokens)})")

    # 4. 写 Markdown 报告
    lines = []
    lines.append("# RTS Comparison Report\n")
    lines.append("## 1. Manual RTS vs Expanded RTS (clean)\n")
    lines.append(f"- 手工 RTS 大小: **{len(manual_tokens)}**")
    lines.append(f"- 扩展+清洗 RTS 大小: **{len(expanded_tokens)}**")
    lines.append(f"- 交集大小: **{len(inter_manual_expanded)}**")
    lines.append(f"- 手工独有 token 数: **{len(manual_only)}**")
    lines.append(f"- 扩展独有 token 数: **{len(expanded_only)}**\n")

    if manual_only:
        sample = ", ".join(sorted(list(manual_only))[:20])
        lines.append(f"- 手工独有 token 示例 (最多 20 个): {sample}\n")
    if expanded_only:
        sample = ", ".join(sorted(list(expanded_only))[:20])
        lines.append(f"- 扩展独有 token 示例 (最多 20 个): {sample}\n")

    lines.append("## 2. Text vs Vision Candidate Tokens（若提供 CSV）\n")
    if text_tokens or vision_tokens:
        lines.append(f"- 文本候选 token 数: **{len(text_tokens)}**")
        lines.append(f"- 图文候选 token 数: **{len(vision_tokens)}**")
        lines.append(f"- 交集大小: **{len(inter_tv)}**")
        lines.append(f"- 文本独有 token 数: **{len(text_only)}**")
        lines.append(f"- 图文独有 token 数: **{len(vision_only)}**\n")

        if text_only:
            sample = ", ".join(sorted(list(text_only))[:20])
            lines.append(f"- 文本独有 token 示例 (最多 20 个): {sample}\n")
        if vision_only:
            sample = ", ".join(sorted(list(vision_only))[:20])
            lines.append(f"- 图文独有 token 示例 (最多 20 个): {sample}\n")
    else:
        lines.append("- 未提供 candidate_text_csv / candidate_vision_csv，跳过此部分。\n")

    lines.append("## 3. Final RTS 选择口径\n")
    lines.append(
        "- 本实验中，最终 RTS 采用 **扩展+清洗后的 RTS** 作为主集合，"
        "以保证在保留手工核心拒绝表述的前提下，引入部分通过 hidden state → logits 得到的扩展拒绝词。"
    )
    lines.append(
        f"- 最终 RTS 已输出至 `{out_final_path}`，后续所有 λ / direction / steer 实验统一加载此文件。"
    )

    out_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote report markdown to: {out_md_path}")


if __name__ == "__main__":
    main()
