
"""
zero_shot_eval.py

python src/scripts/zero_shot_eval.py \
       /root/tkgc_data/share/WIKIDATA12k \
       /root/auto-tmp/qwen7b-ft-merged \
       --batch 4 \
       --topk 5 \
       --temp 0.7 \
       --print_first 3
"""
from __future__ import annotations
import argparse, os, re, sys, time
from pathlib import Path
from typing import List, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR  = THIS_DIR.parent
sys.path.append(str(SRC_DIR))

from prompt.prompt_builder import PromptBuilder, Triple
from model.inference_engine import LLMInferenceEngine, SamplingConfig

import csv, json, datetime
from dataclasses import asdict, dataclass

# ---- (1) aeIoU: 1D "affinity-enhanced IoU" ----
def affinity_enhanced_iou(pred: Tuple[int | None, int | None],
                          gold: Tuple[int | None, int | None]) -> float:
    (ps, pe), (gs, ge) = pred, gold
    if None in (ps, pe, gs, ge):
        return 0.0
    if ps > pe: ps, pe = pe, ps
    if gs > ge: gs, ge = ge, gs
    inter = max(0, min(pe, ge) - max(ps, gs) + 1)
    hull  = max(pe, ge) - min(ps, gs) + 1
    if hull <= 0:
        return 0.0
    iou = inter / hull if hull else 0.0
    if inter > 0:
        return iou  # overlap: same as IoU
    # disjoint: use normalized "closeness" 1 - gap/|C|
    gap = max(gs - pe, ps - ge)  # positive distance between disjoint intervals
    affinity = max(0.0, 1.0 - gap / hull)
    return affinity

# ---- (2) endpoint absolute error ----
def endpoint_abs_errors(
    pred: Tuple[int | None, int | None],
    gold: Tuple[int | None, int | None]
) -> Tuple[float | None, float | None, float | None]:
    (ps, pe), (gs, ge) = pred, gold
    if None in (gs,):
        ae_s = None
    else:
        ae_s = None if ps is None else abs(ps - gs)
    if None in (ge,):
        ae_e = None
    else:
        ae_e = None if pe is None else abs(pe - ge)
    if ae_s is None or ae_e is None:
        ae_avg = None
    else:
        ae_avg = 0.5 * (ae_s + ae_e)
    return ae_s, ae_e, ae_avg

# ---- (3) simple IoU for reference (overlap only) ----
def interval_iou(pred, gold) -> float:
    (ps, pe), (gs, ge) = pred, gold
    if None in (ps, pe, gs, ge):
        return 0.0
    inter = max(0, min(pe, ge) - max(ps, gs) + 1)
    union = max(pe, ge) - min(ps, gs) + 1
    return inter / union if union else 0.0

# ---- (4) per-example record ----
@dataclass
class ExampleLog:
    idx: int
    s: int
    r: int
    o: int
    gold_start: int | None
    gold_end: int | None
    top1_start: int | None
    top1_end: int | None
    giou_top1: float
    aeiou_top1: float
    ae_start_top1: float | None
    ae_end_top1: float | None
    ae_avg_top1: float | None
    giou_bestk: float
    aeiou_bestk: float
    ae_start_bestk: float | None
    ae_end_bestk: float | None
    ae_avg_bestk: float | None
    matched_rank: int | None
    cand1_text: str | None
    cand1_start: int | None
    cand1_end: int | None
    cand2_text: str | None
    cand2_start: int | None
    cand2_end: int | None
    cand3_text: str | None
    cand3_start: int | None
    cand3_end: int | None


# ---------- year helpers ----------
_DIGIT4 = re.compile(r"(\d{4})")
_SPAN   = re.compile(r"(\d{4})\D{0,10}(\d{4})")

def year_or_none(s: str) -> int | None:
    m = _DIGIT4.search(s)
    return int(m.group(1)) if m else None

def gold_interval(beg: str, end: str) -> Tuple[int | None, int | None]:
    return year_or_none(beg), year_or_none(end)

def parse_interval(text: str) -> Tuple[int | None, int | None] | None:
    text = text.strip()
    m = _SPAN.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _DIGIT4.search(text)
    if m:
        y = int(m.group(1)); return y, y
    return None

def generalized_iou(pred: Tuple[int | None, int | None],
                    gold: Tuple[int | None, int | None]) -> float:
    (ps, pe), (gs, ge) = pred, gold
    if None in (ps, pe, gs, ge):
        return 0.0
    inter = max(0, min(pe, ge) - max(ps, gs) + 1)
    union = max(pe, ge) - min(ps, gs) + 1
    return inter / union if union else 0.0

# ---------- data loader ----------
def load_split(path: Path) -> List[Triple]:
    triples: List[Triple] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s, r, o, ts, te = line.rstrip("\n").split("\t")
            triples.append((int(s), int(r), int(o), ts, te))
    return triples

# ---------- evaluation ----------
def evaluate(
    triples: Sequence[Triple],
    builder: PromptBuilder,
    engine: LLMInferenceEngine,
    batch_size: int,
    top_k: int,
    temp: float,
    print_first: int,
) -> None:
    cfg = SamplingConfig(
        temperature=temp,
        top_p=0.9,
        max_tokens=12,
        n=top_k,
        use_beam_search=(temp == 0.0 and top_k > 1),
        stop=["\n"],
    )

    hits_k = 0
    rr_sum = 0.0
    giou_sum = 0.0
    printed = 0
    total   = len(triples)

    t0 = time.time()
    for idx in range(0, total, batch_size):
        batch = triples[idx : idx + batch_size]
        prompts = [builder.build_prompt([], t) for t in batch]
        outputs = engine.batch_generate(prompts, cfg)   # List[List[str]]

        for t, cand_list in zip(batch, outputs):
            gold_int = gold_interval(t[3], t[4])

            matched_rank = None
            for rank, text in enumerate(cand_list, 1):
                pred_int = parse_interval(text) or (None, None)
                if pred_int == gold_int:
                    matched_rank = rank
                if rank == 1:                    # gIOU 只看 top‑1
                    giou_sum += generalized_iou(pred_int, gold_int)

            if matched_rank is not None:
                hits_k += 1
                rr_sum += 1.0 / matched_rank

        done = min(idx + batch_size, total)

        # ---------- pretty‑print ----------
        if print_first and printed < print_first:
            for prm, cand_list in zip(prompts, outputs):
                q_lines = [ln for ln in prm.splitlines() if "Question" in ln or "predict the yearly" in ln]
                question = q_lines[0] if q_lines else "<no‑question>"
                print("\nPROMPT:", question.strip())
                for i, cand in enumerate(cand_list, 1):
                    print(f"  [{i}] {cand.strip()}")
                printed += 1
                if printed >= print_first:
                    break
        # ----------------------------------

        print(f"[{done}/{total}] processed", end="\r", flush=True)

    dur = time.time() - t0
    print("\n===== ZERO‑SHOT RESULT =====")
    print(f"Hits@{top_k} : {hits_k / total:.4f}")
    print(f"MRR         : {rr_sum / total:.4f}")
    print(f"gIOU@1      : {giou_sum / total:.4f}")
    print(f"Time        : {dur/60:.1f} min")

# ---------- CLI ----------
def main() -> None:
    p = argparse.ArgumentParser("Top‑k zero‑shot evaluator")
    p.add_argument("dataset_dir")
    p.add_argument("model")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--topk",  type=int, default=3)
    p.add_argument("--temp",  type=float, default=0.0)
    p.add_argument("--print_first", type=int, default=100,
                   help="Print first n predictions")
    args = p.parse_args()

    triples = load_split(Path(args.dataset_dir) / "test.txt")
    print(f"Loaded {len(triples):,} test triples")

    builder = PromptBuilder(args.dataset_dir, context_mode="both")
    engine  = LLMInferenceEngine(
        args.model,
        dtype="float16",
        load_kwargs={"download_dir": os.getenv("HF_HOME", "/tmp/hf_cache")},
    )

    evaluate(triples, builder, engine,
             batch_size=args.batch,
             top_k=args.topk,
             temp=args.temp,
             print_first=args.print_first)

if __name__ == "__main__":
    main()
