#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_data.py – create SFT jsonl using *structured* PromptBuilder v5
"""

from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from typing import List, Tuple

# ---------- import PromptBuilder ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # …/kg-TKGC-Project/src
sys.path.append(str(PROJECT_ROOT))                   # make `prompt` importable
from prompt.prompt_builder import PromptBuilder, Triple
# ------------------------------------------

# ---------- simple date helpers ------------
def year(t: str) -> str | None: return t[:4] if t[:4].isdigit() else None
def span(ts: str, te: str) -> str:
    ys, ye = year(ts), year(te)
    if ys and ye and ys == ye: return ys
    if ys and ye: return f"{ys}-{ye}"
    if ys: return f"{ys}-####"
    if ye: return f"####-{ye}"
    return "####-####"

# ---------- optional 1‑hop context ----------
def extract_one_hop(
    s_idx: str, o_idx: str, edges: List[Tuple[str, str, str, str, str]],
    k: int = 3,
) -> List[Triple]:
    ctx = []
    for s, r, o, ts, te in edges:
        if len(ctx) >= k: break
        if s == s_idx and o != o_idx:
            ctx.append((s, r, o, ts, te))
        elif o == s_idx and s != o_idx:
            ctx.append((s, r, o, ts, te))
    return ctx

# ---------- main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split_ratio", type=float, default=0.9)
    ap.add_argument("--add_multihop", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    out  = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # init structured PromptBuilder (desc + triples, 12 ctx)
    pb = PromptBuilder(root, context_mode="both", max_context_triples=12)

    # read all train edges
    edges = [tuple(l.rstrip("\n").split("\t")) for l in (root / "train.txt").open()]
    samples = []
    for s, r, o, ts, te in edges:
        ctx = extract_one_hop(s, o, edges) if args.add_multihop else []
        prompt = pb.build_prompt(ctx, (s, r, o, ts, te))
        target = span(ts, te) + "\n"
        samples.append({"prompt": prompt, "target": target})

    random.shuffle(samples)
    split = int(len(samples) * args.split_ratio)
    with (out / "sft_train.jsonl").open("w") as ft:
        for x in samples[:split]:
            ft.write(json.dumps(x, ensure_ascii=False) + "\n")
    with (out / "sft_valid.jsonl").open("w") as fv:
        for x in samples[split:]:
            fv.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"Saved {split} train / {len(samples)-split} valid → {out}")

if __name__ == "__main__":
    main()
