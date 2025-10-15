#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reval_from_csv.py — strict re-evaluation for TKGC interval predictions.

Key points:
- Always average over TOTAL rows; unparsable predictions => score = 0 (but counted).
- Report parsable_rate, overlap_rate, conditional means (overlap vs disjoint),
  endpoint AE stats, sanity warnings for values out of range.
- Robust Hit@1 (guard NaN), robust year parsing from pred_text.
- Inclusive years (length = end - start + 1).

Usage:
  python reval_from_csv.py runs/yg12k_multihop --out reeval_out
  # 'runs/yg12k_multihop' can be a dir containing test_preds_*.csv or a csv file.
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

YEAR_SPAN = re.compile(r"(-?\d{1,4})\D{0,10}(-?\d{1,4})")
YEAR_ONE  = re.compile(r"(-?\d{1,4})")

# ---------------- metrics (inclusive years) ----------------
def _norm(ps: int, pe: int, gs: int, ge: int):
    """Return (inter_len, union_len, hull_len, gap_len) under discrete closed intervals."""
    if ps > pe: ps, pe = pe, ps
    if gs > ge: gs, ge = ge, gs
    # raw overlap without +1
    raw = min(pe, ge) - max(ps, gs)
    inter = max(0, raw + 1)  # discrete length of intersection
    len_p = pe - ps + 1
    len_g = ge - gs + 1
    union = len_p + len_g - inter
    hull = max(pe, ge) - min(ps, gs) + 1
    # "true" hole inside hull (hull - union). 0 if touching.
    gap = max(0, hull - union)
    return inter, union, hull, gap, raw

def iou_1d(pred, gold) -> float:
    ps, pe = pred; gs, ge = gold
    inter, union, hull, _, _ = _norm(ps, pe, gs, ge)
    return (inter / union) if union > 0 else 0.0

def giou_1d(pred, gold) -> float:
    """Scaled gIoU in [0,1]: gIoU' = (1 + gIoU) / 2.
    If you need raw gIoU in [-1,1], compute: 2*giou_1d(...) - 1.
    """
    ps, pe = pred; gs, ge = gold
    inter, union, hull, gap_len, _ = _norm(ps, pe, gs, ge)
    if hull <= 0: return 0.0
    iou = inter / union if union > 0 else 0.0
    giou_raw = iou - (gap_len / hull)         # original gIoU in [-1,1]
    return 0.5 * (1.0 + giou_raw)             # match your earlier code

def aeiou_1d(pred, gold) -> float:
    ps, pe = pred; gs, ge = gold
    inter, union, hull, _, _ = _norm(ps, pe, gs, ge)
    if hull <= 0: return 0.0
    return (inter / union) if inter > 0 else (1.0 / hull)

def gaeiou_1d(pred, gold) -> float:
    """Generalized aeIoU with 'gap+1' smoothing when disjoint,
    exactly mirroring the earlier gaeiou_score implementation.
    """
    ps, pe = pred; gs, ge = gold
    inter, union, hull, _, raw = _norm(ps, pe, gs, ge)
    if hull <= 0: return 0.0
    if inter > 0:
        # when overlapping, denominator hull==union, equals IoU/aeIoU overlap branch
        return inter / union
    # disjoint: numerator = 1 / (abs(raw) + 1); raw < 0 here
    gap_plus1 = max(0, max(ps, gs) - min(pe, ge) + 1)  # == abs(raw)+1 for disjoint
    return (1.0 / (gap_plus1 * hull)) if gap_plus1 > 0 else 0.0

def endpoint_abs_errors(pred:Tuple[int,int]|None, gold:Tuple[int,int]|None):
    if (pred is None) or (gold is None): return None, None
    ps, pe = pred; gs, ge = gold
    return abs(ps - gs), abs(pe - ge)

# ---------------- parsing ----------------
def parse_pred_from_text(text: str|None) -> Optional[Tuple[int,int]]:
    if not text:
        return None
    s = str(text)
    m = YEAR_SPAN.search(s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > b: a, b = b, a
        return a, b
    m1 = YEAR_ONE.search(s)
    if m1:
        y = int(m1.group(1)); return y, y
    return None

def coerce_year(x) -> Optional[int]:
    if x is None: return None
    try:
        if isinstance(x, float) and np.isnan(x):
            return None
        return int(x)
    except Exception:
        return None

def is_intlike(x) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x))

# ---------------- core eval ----------------
def evaluate_file(csv_path: Path, out_dir: Path, method_name: Optional[str]=None) -> Dict:
    df = pd.read_csv(csv_path)

    # Prefer recomputing pred years from text (avoid leakage from earlier runs)
    if "pred_text" in df.columns:
        pred_pair = df["pred_text"].apply(parse_pred_from_text)
        df["pred_start"] = pred_pair.apply(lambda x: x[0] if x else np.nan)
        df["pred_end"]   = pred_pair.apply(lambda x: x[1] if x else np.nan)
    else:
        if not {"pred_start","pred_end"}.issubset(df.columns):
            raise ValueError(f"{csv_path} must contain pred_text or pred_start/pred_end")

    # Coerce to ints (or None)
    for c in ["gold_start","gold_end","pred_start","pred_end"]:
        if c in df.columns:
            df[c] = df[c].apply(coerce_year)

    N = len(df)
    gold_valid = df["gold_start"].apply(is_intlike) & df["gold_end"].apply(is_intlike)
    pred_valid = df["pred_start"].apply(is_intlike) & df["pred_end"].apply(is_intlike)
    both_valid = gold_valid & pred_valid

    # Hit@1 — guard NaN/None before casting
    hits = 0
    for _, row in df.iterrows():
        gs, ge = row.get("gold_start"), row.get("gold_end")
        ps, pe = row.get("pred_start"), row.get("pred_end")
        if is_intlike(gs) and is_intlike(ge) and is_intlike(ps) and is_intlike(pe):
            if int(gs) == int(ps) and int(ge) == int(pe):
                hits += 1
    hit1 = hits / N if N else 0.0

    # Metrics (strict: unparsable -> 0; still counted)
    ious, gious, aei, gaei, overlaps = [], [], [], [], []
    ae_s_list, ae_e_list = [], []
    for _, row in df.iterrows():
        gs, ge = row.get("gold_start"), row.get("gold_end")
        ps, pe = row.get("pred_start"), row.get("pred_end")
        if not (is_intlike(gs) and is_intlike(ge)):
            # invalid gold -> define as zero
            ious.append(0.0); gious.append(0.0); aei.append(0.0); gaei.append(0.0); overlaps.append(False)
            continue
        gs_i, ge_i = int(gs), int(ge)

        if not (is_intlike(ps) and is_intlike(pe)):
            ious.append(0.0); gious.append(0.0); aei.append(0.0); gaei.append(0.0); overlaps.append(False)
            continue
        ps_i, pe_i = int(ps), int(pe)

        inter, union, hull, gap, _ = _norm(ps_i, pe_i, gs_i, ge_i)
        overlaps.append(inter > 0)
        ious.append(iou_1d((ps_i, pe_i), (gs_i, ge_i)))
        gious.append(giou_1d((ps_i, pe_i), (gs_i, ge_i)))
        aei.append(aeiou_1d((ps_i, pe_i), (gs_i, ge_i)))
        gaei.append(gaeiou_1d((ps_i, pe_i), (gs_i, ge_i)))

        # endpoint AE for parsable rows (diagnostic only)
        ae_s, ae_e = endpoint_abs_errors((ps_i, pe_i), (gs_i, ge_i))
        if ae_s is not None: ae_s_list.append(ae_s)
        if ae_e is not None: ae_e_list.append(ae_e)

    # Convert
    ious  = np.asarray(ious, dtype=float)
    gious = np.asarray(gious, dtype=float)
    aei   = np.asarray(aei, dtype=float)
    gaei  = np.asarray(gaei, dtype=float)
    overlaps = np.asarray(overlaps, dtype=bool)

    # Sanity clamp + warn
    for name, arr in [("IoU", ious), ("gIoU", gious), ("aeIoU", aei), ("gaeIoU", gaei)]:
        if np.nanmax(arr) > 1.0000001:
            print(f"[WARN] {csv_path.name}: {name} has values > 1, max={np.nanmax(arr):.6f}")
        arr[arr < -1.0] = -1.0
        arr[arr >  1.0] =  1.0

    # Global means (STRICT: /N)
    iou_mean  = float(np.sum(ious)  / max(1, N))
    giou_mean = float(np.sum(gious) / max(1, N))
    aei_mean  = float(np.sum(aei)   / max(1, N))
    gae_mean  = float(np.sum(gaei)  / max(1, N))

    # Coverage & overlap stats
    parsable = int(np.sum(both_valid))
    parsable_rate = parsable / max(1, N)
    overlap_rate  = float(np.sum(overlaps & both_valid.to_numpy())) / max(1, parsable)

    # Conditional means among parsable rows
    if parsable > 0:
        mask_p  = both_valid.to_numpy()
        mask_ov = overlaps & mask_p
        mask_ds = (~overlaps) & mask_p
        giou_ov_mean = float(np.mean(gious[mask_ov])) if np.any(mask_ov) else float("nan")
        aei_ov_mean  = float(np.mean(aei[mask_ov]))   if np.any(mask_ov) else float("nan")
        gae_ov_mean  = float(np.mean(gaei[mask_ov]))  if np.any(mask_ov) else float("nan")
        giou_ds_mean = float(np.mean(gious[mask_ds])) if np.any(mask_ds) else float("nan")
        aei_ds_mean  = float(np.mean(aei[mask_ds]))   if np.any(mask_ds) else float("nan")
        gae_ds_mean  = float(np.mean(gaei[mask_ds]))  if np.any(mask_ds) else float("nan")
    else:
        giou_ov_mean = aei_ov_mean = gae_ov_mean = float("nan")
        giou_ds_mean = aei_ds_mean = gae_ds_mean = float("nan")

    # Endpoint AE stats（parsable rows）
    def _ae_stats(xs: List[int]):
        if len(xs) == 0:
            return dict(mean=np.nan, std=np.nan, p50=np.nan, p90=np.nan)
        x = np.asarray(xs, dtype=float)
        return dict(
            mean=float(np.mean(x)),
            std=float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            p50=float(np.percentile(x, 50)),
            p90=float(np.percentile(x, 90)),
        )
    ae_start_stats = _ae_stats(ae_s_list)
    ae_end_stats   = _ae_stats(ae_e_list)

    # Optional best-of-k (diagnostic only) if cand*_text present
    bestk = {}
    cand_cols = [c for c in df.columns if c.startswith("cand") and c.endswith("_text")]
    if len(cand_cols) >= 2:
        def best_of_k(row):
            gs, ge = row.get("gold_start"), row.get("gold_end")
            if not (is_intlike(gs) and is_intlike(ge)):
                return 0.0
            best = 0.0
            for c in cand_cols:
                pred = parse_pred_from_text(row.get(c))
                if pred is None: continue
                best = max(best, giou_1d(pred, (int(gs), int(ge))))
            return best
        g_best = df.apply(best_of_k, axis=1).to_numpy()
        bestk["gIoU@bestk_mean"] = float(np.sum(g_best) / max(1, N))

    summary = {
        "file": csv_path.name,
        "method": method_name or (df["method"].iloc[0] if "method" in df.columns else "unknown"),
        "total": N,
        "parsable": parsable,
        "parsable_rate": parsable_rate,
        "hit@1": hit1,
        "IoU@1": iou_mean,
        "gIoU@1": giou_mean,
        "aeIoU@1": aei_mean,
        "gaeIoU@1": gae_mean,
        "overlap_rate": overlap_rate,
        "gIoU_mean|overlap": giou_ov_mean,
        "aeIoU_mean|overlap": aei_ov_mean,
        "gaeIoU_mean|overlap": gae_ov_mean,
        "gIoU_mean|disjoint": giou_ds_mean,
        "aeIoU_mean|disjoint": aei_ds_mean,
        "gaeIoU_mean|disjoint": gae_ds_mean,
        "AE_start": ae_start_stats,
        "AE_end": ae_end_stats,
    }
    summary.update(bestk)

    # pretty print
    print(f"\n== {csv_path.name} ==")
    for k in ["total","parsable","parsable_rate","hit@1","IoU@1","gIoU@1","aeIoU@1","gaeIoU@1",
              "overlap_rate","gIoU_mean|overlap","aeIoU_mean|overlap","gaeIoU_mean|overlap",
              "gIoU_mean|disjoint","aeIoU_mean|disjoint","gaeIoU_mean|disjoint"]:
        v = summary[k]
        if isinstance(v, float):
            print(f"{k:<22} {v:.6f}")
        else:
            print(f"{k:<22} {v}")
    print(f"AE_start stats         {summary['AE_start']}")
    print(f"AE_end   stats         {summary['AE_end']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"reeval_strict_{csv_path.stem}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_json}")
    return summary

# ---------------- batch driver ----------------
def _expand_inputs(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        P = Path(p)
        if P.is_dir():
            out += sorted(P.glob("test_preds_*.csv"))
        elif P.suffix.lower() == ".csv":
            out.append(P)
    if not out:
        raise FileNotFoundError("No CSVs found. Pass CSV files or a dir containing test_preds_*.csv")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="CSV files or directories (contain test_preds_*.csv)")
    ap.add_argument("--out", default="reeval_out", type=str, help="output directory for JSON/CSV summaries")
    args = ap.parse_args()

    out_dir = Path(args.out)
    files = _expand_inputs(args.inputs)
    all_summ = []
    for f in files:
        all_summ.append(evaluate_file(f, out_dir))

    # aggregate table
    df = pd.DataFrame(all_summ)
    df.to_csv(out_dir / "reeval_summary_all.csv", index=False)
    print(f"[saved] {out_dir / 'reeval_summary_all.csv'}")

if __name__ == "__main__":
    main()
