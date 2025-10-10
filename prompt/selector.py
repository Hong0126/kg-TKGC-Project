# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import math, re
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass

from .prompt_builder import Triple  # type: ignore

# ===== Adjustable params =====
H1_PER_ANCHOR = 12
H2_TOTAL      = 8 
REL_CAP       = 3
TAU_YEARS     = 15.0 
SPLIT_FILES   = ("train.txt", "valid.txt", "test.txt")

# ===== Internal cache index =====
_ADJ: Dict[int, List[Triple]] = {} 
_DEG: Dict[int, int] = {} 
_REL_FREQ: Dict[int, int] = {} 
_REL_IDF: Dict[int, float] = {} 
_INDEX_BUILT = False

_DIGIT4 = re.compile(r"(\d{4})")
def _year(s: str) -> Optional[int]:
    if not s: return None
    m = _DIGIT4.search(s)
    return int(m.group(1)) if m else None

def _mid_year(ts: str, te: str) -> Optional[float]:
    ys, ye = _year(ts), _year(te)
    if ys is not None and ye is not None: return 0.5*(ys+ye)
    if ys is not None: return float(ys)
    if ye is not None: return float(ye)
    return None

def _build_index(kg_root: Path) -> None:
    global _ADJ, _DEG, _REL_FREQ, _REL_IDF, _INDEX_BUILT
    if _INDEX_BUILT:
        return
    adj = defaultdict(list)
    deg = Counter()
    rfreq = Counter()
    for fname in SPLIT_FILES:
        p = kg_root / fname
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line: continue
                s, r, o, ts, te = line.split("\t")
                s, r, o = int(s), int(r), int(o)
                tri: Triple = (s, r, o, ts, te)
                adj[s].append(tri); adj[o].append(tri)
                deg[s] += 1; deg[o] += 1
                rfreq[r] += 1
    N = sum(rfreq.values()) or 1
    rel_idf = {r: math.log((N + 1) / (c + 1)) for r, c in rfreq.items()}

    _ADJ = dict(adj)
    _DEG = dict(deg)
    _REL_FREQ = dict(rfreq)
    _REL_IDF = rel_idf
    _INDEX_BUILT = True

def _score_edge_for_anchor(anchor: int, tri: Triple, anchor_med: Optional[float]) -> float:
    s, r, o, ts, te = tri
    other = o if s == anchor else s
    ridf = _REL_IDF.get(r, 0.0)
    mid  = _mid_year(ts, te)
    if (mid is not None) and (anchor_med is not None):
        t_aff = math.exp(-abs(mid - anchor_med) / TAU_YEARS)
    else:
        t_aff = 0.5
    cent = 1.0 / math.sqrt(1.0 + _DEG.get(other, 0))   # 反 hub
    return 0.5*ridf + 0.3*t_aff + 0.2*cent

def _rank_1hop(anchor: int) -> List[Tuple[float, Triple]]:
    edges = _ADJ.get(anchor, [])
    if not edges:
        return []
    mids = [m for (_,_,_,ts,te) in edges if (m := _mid_year(ts, te)) is not None]
    med = float(np.median(mids)) if mids else None
    scored = [( _score_edge_for_anchor(anchor, tri, med), tri ) for tri in edges]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

def _cap_by_relation(scored: List[Tuple[float, Triple]], cap: int) -> List[Triple]:
    out: List[Triple] = []
    per_rel = Counter()
    for _, tri in scored:
        r = int(tri[1])
        if per_rel[r] >= REL_CAP:
            continue
        out.append(tri)
        per_rel[r] += 1
        if len(out) >= cap:
            break
    return out

def _pick_2hop(one_hop: List[Triple], anchors: set[int], total_cap: int) -> List[Triple]:
    cand: List[Tuple[float, Triple]] = []
    used_mid: set[int] = set()
    for (s, r, o, ts, te) in one_hop:
        if s in anchors and o not in anchors:
            mid = o
        elif o in anchors and s not in anchors:
            mid = s
        else:
            continue
        if mid in used_mid:
            continue
        for (ss, rr, oo, tss, tee) in _ADJ.get(mid, []):
            if (ss, rr, oo, tss, tee) in one_hop:
                continue
            other_end = oo if ss == mid else ss
            base  = _REL_IDF.get(rr, 0.0) + 0.2*(1.0 / math.sqrt(1.0 + _DEG.get(other_end, 0)))
            bonus = 0.5 if other_end in anchors else 0.0
            score = base + bonus
            cand.append((score, (ss, rr, oo, tss, tee)))
        used_mid.add(mid)

    cand.sort(key=lambda x: x[0], reverse=True)
    out: List[Triple] = []
    taken_mid: set[int] = set()
    for _, tri in cand:
        ss, rr, oo, tss, tee = tri
        if ss in anchors and oo not in anchors:
            mid = oo
        elif oo in anchors and ss not in anchors:
            mid = ss
        else:
            mid = ss
        if mid in taken_mid:
            continue
        out.append(tri)
        taken_mid.add(mid)
        if len(out) >= total_cap:
            break
    return out

# ===================== Stage B Dynamic Pruning=====================

@dataclass(frozen=True)
class CandEdge:
    s: int; r: int; o: int; ts: str; te: str
    score_static: float
    mid_year: Optional[float]
    other_deg: int
    key: Tuple[int,int,int,str,str]

def enumerate_candidates(
    kg_root: Path, s: int, o: int,
    hop_mode: str = "12",
    h1_per_anchor: int = H1_PER_ANCHOR,
    h2_total: int = H2_TOTAL,
) -> List[CandEdge]:
    _build_index(kg_root)
    if hop_mode == "0":
        return []

    anchors = {int(s), int(o)}
    s_ranked = _rank_1hop(int(s))
    o_ranked = _rank_1hop(int(o))
    s_1hop   = _cap_by_relation(s_ranked, h1_per_anchor)
    o_1hop   = _cap_by_relation(o_ranked, h1_per_anchor)

    seen = set()
    one_hop: list[Triple] = []
    for tri in (s_1hop + o_1hop):
        key = (tri[0], tri[1], tri[2], tri[3], tri[4])
        if key in seen: 
            continue
        if tri[0] in anchors or tri[2] in anchors:
            one_hop.append(tri); seen.add(key)

    if hop_mode == "1":
        two_hop: list[Triple] = []
    else:
        two_hop = _pick_2hop(one_hop, anchors, total_cap=h2_total)

    def _static_score(tri: Triple) -> float:
        sc1 = _score_edge_for_anchor(int(s), tri,
                anchor_med=np.median([m for (_,_,_,ts,te) in _ADJ.get(int(s), []) if (m:=_mid_year(ts,te)) is not None]) 
                if _ADJ.get(int(s)) else None)
        sc2 = _score_edge_for_anchor(int(o), tri,
                anchor_med=np.median([m for (_,_,_,ts,te) in _ADJ.get(int(o), []) if (m:=_mid_year(ts,te)) is not None]) 
                if _ADJ.get(int(o)) else None)
        return float(max(sc1, sc2))

    cands: List[CandEdge] = []
    for tri in (one_hop + two_hop):
        ss, rr, oo, ts, te = tri
        other = oo if ss in anchors else ss
        mid = _mid_year(ts, te)
        cands.append(CandEdge(
            s=ss, r=rr, o=oo, ts=ts, te=te,
            score_static=_static_score(tri),
            mid_year=mid,
            other_deg=_DEG.get(other, 0),
            key=(ss, rr, oo, ts, te),
        ))
    cands.sort(key=lambda x: x.score_static, reverse=True)
    return cands

CostFn = Callable[[CandEdge], int]
GainFn = Callable[[CandEdge, List[CandEdge], Optional[Tuple[int,int]], Counter, Counter], float]

def _default_cost_fn(edge: CandEdge) -> int:
    return 1

def _overlap_ratio(edge: CandEdge, target: Tuple[int,int]) -> float:
    ys, ye = _year(edge.ts), _year(edge.te)
    if ys is None and ye is None:
        return 0.5 
    if ys is None: ys = ye
    if ye is None: ye = ys
    a, b = int(min(ys, ye)), int(max(ys, ye))
    c, d = int(min(target[0], target[1])), int(max(target[0], target[1]))
    inter = max(0, min(b, d) - max(a, c) + 1)
    span  = max(1, max(b, d) - min(a, c) + 1)
    return inter / span

def _default_gain_fn(
    edge: CandEdge, selected: List[CandEdge],
    target_interval: Optional[Tuple[int,int]],
    rel_hist: Counter, mid_hist: Counter
) -> float:
    if target_interval is not None:
        g_time = _overlap_ratio(edge, target_interval)
    elif edge.mid_year is not None and selected:
        meds = [e.mid_year for e in selected if e.mid_year is not None]
        if meds:
            g_time = math.exp(-abs(edge.mid_year - float(np.median(meds))) / TAU_YEARS)
        else:
            g_time = 0.5
    else:
        g_time = 0.5

    g_cent = 1.0 / math.sqrt(1.0 + edge.other_deg)

    div_rel = 1.0 / (1.0 + rel_hist[edge.r])
    mid = edge.o if edge.s in (selected[0].s if selected else -1, ) else edge.s
    div_mid = 1.0 / (1.0 + mid_hist[mid])

    return 0.4*edge.score_static + 0.35*g_time + 0.15*g_cent + 0.10*(div_rel*div_mid)

def select_context_dynamic(
    kg_root: Path, s: int, o: int,
    hop_mode: str = "12",
    budget: int = 16,
    budget_type: str = "edges",            # "edges" | "tokens"
    target_interval: Optional[Tuple[int,int]] = None,  # (t_start, t_end)
    cost_fn: CostFn = _default_cost_fn,
    gain_fn: GainFn = _default_gain_fn,
    h1_per_anchor: int = H1_PER_ANCHOR,
    h2_total: int = H2_TOTAL,
    rel_cap: int = REL_CAP,
) -> Tuple[List[Triple], Dict[str, object]]:
    cands = enumerate_candidates(kg_root, s, o, hop_mode, h1_per_anchor, h2_total)
    selected: List[CandEdge] = []
    rel_hist, mid_hist = Counter(), Counter()

    def _can_take(e: CandEdge) -> bool:
        return rel_hist[e.r] < rel_cap

    used_budget = 0
    remaining = set(cands)

    while remaining:
        scored = []
        for e in list(remaining):
            if not _can_take(e):
                remaining.discard(e)
                continue
            gain = gain_fn(e, selected, target_interval, rel_hist, mid_hist)
            cost = cost_fn(e)
            if cost <= 0:
                cost = 1
            u = gain / float(cost)
            scored.append((u, gain, cost, e))
        if not scored:
            break
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_u, best_gain, best_cost, best_e = scored[0]

        # 预算检查
        if budget_type == "edges":
            if used_budget + 1 > budget:
                break
            used_budget += 1
        else:  # tokens
            if used_budget + best_cost > budget:
                remaining.discard(best_e)
                continue
            used_budget += best_cost

        selected.append(best_e)
        rel_hist[best_e.r] += 1
        mid_node = best_e.o if best_e.s in {s, o} else best_e.s
        mid_hist[mid_node] += 1
        remaining.discard(best_e)

    # 输出
    triples: List[Triple] = [(e.s, e.r, e.o, e.ts, e.te) for e in selected]
    stats = {
        "budget_type": budget_type,
        "budget_used": used_budget,
        "num_selected": len(triples),
        "num_candidates": len(cands),
        "rel_hist": dict(rel_hist),
        "mid_hist_size": len(mid_hist),
        "target_interval": target_interval,
        "hop_mode": hop_mode,
    }
    return triples, stats


def select_context(kg_root: Path, s: int, o: int,
                   hop_mode: str = "12",            # "0" | "1" | "12"
                   h1_per_anchor: int = H1_PER_ANCHOR,
                   h2_total: int = H2_TOTAL) -> tuple[list[Triple], int, int]:
    _build_index(kg_root)

    if hop_mode == "0":
        return [], 0, 0

    anchors = {int(s), int(o)}
    s_ranked = _rank_1hop(int(s))
    o_ranked = _rank_1hop(int(o))
    s_1hop   = _cap_by_relation(s_ranked, h1_per_anchor)
    o_1hop   = _cap_by_relation(o_ranked, h1_per_anchor)

    seen = set()
    one_hop: list[Triple] = []
    for tri in (s_1hop + o_1hop):
        key = (tri[0], tri[1], tri[2], tri[3], tri[4])
        if key in seen:
            continue
        if tri[0] in anchors or tri[2] in anchors:
            one_hop.append(tri); seen.add(key)

    if hop_mode == "1":
        return one_hop, len(one_hop), 0

    two_hop = _pick_2hop(one_hop, anchors, total_cap=h2_total)
    return one_hop + two_hop, len(one_hop), len(two_hop)
