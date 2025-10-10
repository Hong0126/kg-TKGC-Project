from __future__ import annotations
import argparse, json, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------- regex ----------------
YEAR_SPAN = re.compile(r"(-?\d{1,4})\D{0,10}(-?\d{1,4})")
YEAR_ONE  = re.compile(r"(-?\d{1,4})")

# ---------------- metrics (inclusive years) ----------------
def _norm(ps: int, pe: int, gs: int, ge: int):
    """Return (inter_len, union_len, hull_len, gap_len, raw_overlap) under discrete closed intervals."""
    if ps > pe: ps, pe = pe, ps
    if gs > ge: gs, ge = ge, gs
    raw = min(pe, ge) - max(ps, gs)          # raw overlap without +1
    inter = max(0, raw + 1)                  # inclusive length
    len_p = pe - ps + 1
    len_g = ge - gs + 1
    union = len_p + len_g - inter
    hull = max(pe, ge) - min(ps, gs) + 1
    gap  = max(0, hull - union)              # 0 if touching
    return inter, union, hull, gap, raw

def iou_1d(pred, gold) -> float:
    ps, pe = pred; gs, ge = gold
    inter, union, *_ = _norm(ps, pe, gs, ge)
    return (inter / union) if union > 0 else 0.0

def giou_1d(pred, gold) -> float:
    """Scaled gIoU in [0,1]: gIoU' = (1 + gIoU_raw) / 2."""
    ps, pe = pred; gs, ge = gold
    inter, union, hull, gap, _ = _norm(ps, pe, gs, ge)
    if hull <= 0: return 0.0
    iou = inter / union if union > 0 else 0.0
    giou_raw = iou - (gap / hull)            # [-1, 1]
    return 0.5 * (1.0 + giou_raw)

def aeiou_1d(pred, gold) -> float:
    ps, pe = pred; gs, ge = gold
    inter, union, hull, *_ = _norm(ps, pe, gs, ge)
    if hull <= 0: return 0.0
    return (inter / union) if inter > 0 else (1.0 / hull)

def gaeiou_1d(pred, gold) -> float:
    """Generalized aeIoU with 'gap+1' smoothing when disjoint."""
    ps, pe = pred; gs, ge = gold
    inter, union, hull, _, raw = _norm(ps, pe, gs, ge)
    if hull <= 0: return 0.0
    if inter > 0:
        return inter / union
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

# ---------------- I/O ----------------
def read_csv_with_fallback(csv_path: Path, delimiter: str = ",") -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, sep=delimiter)
    except Exception as e:
        print(f"[WARN] read_csv(sep='{delimiter}') failed → fallback to auto: {e}", file=sys.stderr)
        return pd.read_csv(csv_path)  # pandas auto

def prepare_df_for_eval(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Recompute pred_start/pred_end from text if present (avoid leakage). Coerce years."""
    if "pred_text" in df.columns:
        pred_pair = df["pred_text"].apply(parse_pred_from_text)
        df["pred_start"] = pred_pair.apply(lambda x: x[0] if x else np.nan)
        df["pred_end"]   = pred_pair.apply(lambda x: x[1] if x else np.nan)
    else:
        if not {"pred_start","pred_end"}.issubset(df.columns):
            raise ValueError(f"{csv_path} must contain pred_text or pred_start/pred_end")

    for c in ["gold_start","gold_end","pred_start","pred_end"]:
        if c in df.columns:
            df[c] = df[c].apply(coerce_year)

    # sanity: drop rows without gold
    mask_gold_ok = df["gold_start"].apply(is_intlike) & df["gold_end"].apply(is_intlike)
    if not mask_gold_ok.all():
        drop_n = int((~mask_gold_ok).sum())
        if drop_n > 0:
            print(f"[INFO] dropping {drop_n} rows without gold years.", file=sys.stderr)
            df = df[mask_gold_ok].reset_index(drop=True)
    return df

# ---------------- strict evaluation on a DataFrame ----------------
def evaluate_df(df: pd.DataFrame, method_name: Optional[str]=None) -> Dict:
    N = len(df)

    # Hit@1
    hits = 0
    for _, row in df.iterrows():
        gs, ge = row.get("gold_start"), row.get("gold_end")
        ps, pe = row.get("pred_start"), row.get("pred_end")
        if is_intlike(gs) and is_intlike(ge) and is_intlike(ps) and is_intlike(pe):
            if int(gs) == int(ps) and int(ge) == int(pe):
                hits += 1
    hit1 = hits / N if N else 0.0

    # Metrics (strict: unparsable -> 0)
    ious, gious, aei, gae, overlaps = [], [], [], [], []
    ae_s_list, ae_e_list = [], []
    for _, row in df.iterrows():
        gs, ge = row.get("gold_start"), row.get("gold_end")
        ps, pe = row.get("pred_start"), row.get("pred_end")

        if not (is_intlike(gs) and is_intlike(ge)):
            ious.append(0.0); gious.append(0.0); aei.append(0.0); gae.append(0.0); overlaps.append(False)
            continue

        if not (is_intlike(ps) and is_intlike(pe)):
            ious.append(0.0); gious.append(0.0); aei.append(0.0); gae.append(0.0); overlaps.append(False)
            continue

        gs_i, ge_i = int(gs), int(ge)
        ps_i, pe_i = int(ps), int(pe)

        inter, union, hull, gap, _ = _norm(ps_i, pe_i, gs_i, ge_i)
        overlaps.append(inter > 0)
        ious.append(iou_1d((ps_i, pe_i), (gs_i, ge_i)))
        gious.append(giou_1d((ps_i, pe_i), (gs_i, ge_i)))
        aei.append(aeiou_1d((ps_i, pe_i), (gs_i, ge_i)))
        gae.append(gaeiou_1d((ps_i, pe_i), (gs_i, ge_i)))

        ae_s, ae_e = endpoint_abs_errors((ps_i, pe_i), (gs_i, ge_i))
        if ae_s is not None: ae_s_list.append(ae_s)
        if ae_e is not None: ae_e_list.append(ae_e)

    ious  = np.asarray(ious, dtype=float)
    gious = np.asarray(gious, dtype=float)
    aei   = np.asarray(aei, dtype=float)
    gae   = np.asarray(gae, dtype=float)
    overlaps = np.asarray(overlaps, dtype=bool)

    for name, arr in [("IoU", ious), ("gIoU", gious), ("aeIoU", aei), ("gaeIoU", gae)]:
        if arr.size and np.nanmax(arr) > 1.0000001:
            print(f"[WARN] {method_name or ''}: {name} has values > 1, max={np.nanmax(arr):.6f}")
        arr[arr < -1.0] = -1.0
        arr[arr >  1.0] =  1.0

    # means over TOTAL rows
    iou_mean  = float(np.sum(ious) / max(1, N))
    giou_mean = float(np.sum(gious) / max(1, N))
    aei_mean  = float(np.sum(aei)   / max(1, N))
    gae_mean  = float(np.sum(gae)   / max(1, N))

    gold_valid = df["gold_start"].apply(is_intlike) & df["gold_end"].apply(is_intlike)
    pred_valid = df["pred_start"].apply(is_intlike) & df["pred_end"].apply(is_intlike)
    both_valid = gold_valid & pred_valid
    parsable   = int(np.sum(both_valid))
    parsable_rate = parsable / max(1, N)
    overlap_rate  = float(np.sum(overlaps & both_valid.to_numpy())) / max(1, parsable) if parsable>0 else 0.0

    if parsable > 0:
        mask_p  = both_valid.to_numpy()
        mask_ov = overlaps & mask_p
        mask_ds = (~overlaps) & mask_p
        giou_ov_mean = float(np.mean(gious[mask_ov])) if np.any(mask_ov) else float("nan")
        aei_ov_mean  = float(np.mean(aei[mask_ov]))   if np.any(mask_ov) else float("nan")
        gae_ov_mean  = float(np.mean(gae[mask_ov]))   if np.any(mask_ov) else float("nan")
        giou_ds_mean = float(np.mean(gious[mask_ds])) if np.any(mask_ds) else float("nan")
        aei_ds_mean  = float(np.mean(aei[mask_ds]))   if np.any(mask_ds) else float("nan")
        gae_ds_mean  = float(np.mean(gae[mask_ds]))   if np.any(mask_ds) else float("nan")
    else:
        giou_ov_mean = aei_ov_mean = gae_ov_mean = float("nan")
        giou_ds_mean = aei_ds_mean = gae_ds_mean = float("nan")

    return {
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
    }

# ---------------- inductive split helpers ----------------
@dataclass
class Seen:
    ents: set[int]
    s_r:  set[tuple[int,int]]
    r_o:  set[tuple[int,int]]

def read_train_seen(train_file: Path) -> Seen:
    ents: set[int] = set()
    s_r:  set[tuple[int,int]] = set()
    r_o:  set[tuple[int,int]] = set()
    with train_file.open("r", encoding="utf-8") as f:
        for l in f:
            if not l.strip(): continue
            parts = l.rstrip("\n").split("\t")
            if len(parts) < 3: continue
            try:
                s, r, o = int(parts[0]), int(parts[1]), int(parts[2])
            except Exception:
                continue
            ents.update([s, o])
            s_r.add((s, r))
            r_o.add((r, o))
    return Seen(ents, s_r, r_o)

def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first existing column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def mark_inductive(df_raw: pd.DataFrame, seen: Seen, mode: str = "strict") -> pd.Series:
    # try multiple aliases
    s_col = pick_col(df_raw, ["subject_id","s","sid","subj","head","h","s_id"])
    r_col = pick_col(df_raw, ["rel_id","r","rel","relation","p","r_id"])
    o_col = pick_col(df_raw, ["object_id","o","oid","obj","tail","t","o_id"])

    n = len(df_raw)
    if s_col is None or o_col is None:
        # 仍然返回全 False，但打印清晰提示，避免“看起来全是 transductive”的错觉
        print("[WARN] inductive split disabled: cannot find subject/object columns in CSV.",
              "Expected one of:",
              "\n  subject:", ["subject_id","s","sid","subj","head","h","s_id"],
              "\n  object :", ["object_id","o","oid","obj","tail","t","o_id"])
        return pd.Series([False] * n, index=df_raw.index)

    # numeric coercion（字符串 id 也能转）
    s = pd.to_numeric(df_raw[s_col], errors="coerce").fillna(-1).astype(int)
    o = pd.to_numeric(df_raw[o_col], errors="coerce").fillna(-1).astype(int)

    # strict: entity 未在训练出现
    strict_mask = (~s.isin(seen.ents)) | (~o.isin(seen.ents))

    if mode == "strict":
        return strict_mask

    # relaxed 还考虑 (s,r) / (r,o) 是否见过；若没关系列，就等价于 strict
    if r_col is None:
        return strict_mask

    r = pd.to_numeric(df_raw[r_col], errors="coerce").fillna(-1).astype(int)
    sr_unseen = pd.Series([ (sv, rv) not in seen.s_r for sv, rv in zip(s.tolist(), r.tolist()) ],
                          index=df_raw.index)
    ro_unseen = pd.Series([ (rv, ov) not in seen.r_o for rv, ov in zip(r.tolist(), o.tolist()) ],
                          index=df_raw.index)

    return strict_mask | sr_unseen | ro_unseen


# ---------------- pretty printing ----------------
def _print_reval_block(name: str, summ: Dict):
    print(f"\n== {name} ==")
    for k in ["total","parsable","parsable_rate","hit@1","IoU@1","gIoU@1","aeIoU@1","gaeIoU@1",
              "overlap_rate","gIoU_mean|overlap","aeIoU_mean|overlap","gaeIoU_mean|overlap",
              "gIoU_mean|disjoint","aeIoU_mean|disjoint","gaeIoU_mean|disjoint"]:
        v = summ.get(k, np.nan)
        if isinstance(v, float):
            print(f"{k:<22} {v:.6f}")
        else:
            print(f"{k:<22} {v}")

def _print_tri_lines(overall: Dict, induct: Dict, trans: Dict):
    def _mkline(lbl: str, d: Dict):
        n = d.get("total", 0)
        hits = d.get("hit@1", 0.0)
        giou = d.get("gIoU@1", 0.0)
        aei  = d.get("aeIoU@1", 0.0)
        gae  = d.get("gaeIoU@1", 0.0)
        return f"{lbl:<14} n={n:5d}  Hits@1={hits:.4f}  MRR={hits:.4f}  gIOU@1={giou:.4f}  aeIoU@1={aei:.4f}  gaeIoU@1={gae:.4f}"
    print("\n===== SCORES =====")
    print(_mkline("overall",      overall))
    print(_mkline("inductive",    induct))
    print(_mkline("transductive", trans))

# ---------------- per-file driver ----------------
def evaluate_csv(csv_path: Path, out_dir: Path, delimiter: str, seen: Seen, inductive_mode: str):
    df_raw = read_csv_with_fallback(csv_path, delimiter=delimiter)
    df     = prepare_df_for_eval(df_raw.copy(), csv_path)

    # overall
    overall = evaluate_df(df, method_name=csv_path.name)
    _print_reval_block(csv_path.name, overall)

    # split masks from raw ids
    ind_mask = mark_inductive(df_raw, seen, mode=inductive_mode)
    ind_mask = ind_mask.reindex(df.index, fill_value=False)

    df_ind   = df[ind_mask]
    df_trans = df[~ind_mask]

    induct  = evaluate_df(df_ind,  method_name=f"{csv_path.name} [inductive]")
    trans   = evaluate_df(df_trans, method_name=f"{csv_path.name} [transductive]")

    _print_tri_lines(overall, induct, trans)

    # save json
    summary = {
        "file": csv_path.name,
        "inductive_mode": inductive_mode,
        "overall": overall,
        "inductive": induct,
        "transductive": trans,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"score_{csv_path.stem}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_json}")
    return summary

# ---------------- batch helper ----------------
def expand_inputs(p: Path) -> List[Path]:
    if p.is_dir():
        files = sorted(p.glob("test_preds_*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSVs found under dir: {p}")
        return files
    if p.suffix.lower() == ".csv":
        return [p]
    raise FileNotFoundError(f"Input must be a CSV file or a directory: {p}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True, help="CSV file or a directory containing test_preds_*.csv")
    ap.add_argument("--train_file", required=True, help="train.txt (s r o t1 t2) to derive seen entities/pairs")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default ',')")
    ap.add_argument("--inductive_mode", default="strict", choices=["strict","relaxed"],
                    help="strict: s/o unseen in train; relaxed: strict OR (s,r) unseen OR (r,o) unseen")
    args = ap.parse_args()

    in_path = Path(args.pred_file)
    out_dir = Path(args.out_dir)
    seen    = read_train_seen(Path(args.train_file))

    files = expand_inputs(in_path)
    all_summ = []
    for f in files:
        all_summ.append(evaluate_csv(f, out_dir, args.delimiter, seen, args.inductive_mode))

    df = pd.DataFrame(all_summ)
    df.to_csv(out_dir / "score_summary_all.csv", index=False)
    print(f"[saved] {out_dir / 'score_summary_all.csv'}")

if __name__ == "__main__":
    main()
