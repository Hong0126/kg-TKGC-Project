from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

EntityId = Union[int, str]
RelId    = Union[int, str]
Triple   = Tuple[EntityId, RelId, EntityId, str, str]  # s, r, o, t1, t2

# ----------------------------- selector config -----------------------------
@dataclass
class SelectorConfig:
    """
    Configuration for invoking the graph-context selector.

    mode:
        - "dynamic": use Stage-B dynamic pruning (greedy, budget-aware)
        - "static":  use legacy static selection (0/1/1+2 hop)
        - "none":    do not fetch graph context (only descriptions)
    hop_mode: "0" | "1" | "12"
    budget: total budget for selection (edge-count or token-count)
    budget_type: "edges" or "tokens"
    target_interval: optional (t_start, t_end) prior for time-aware gains
    cost_fn: optional callable for token-cost estimation per edge (dynamic)
    gain_fn: optional callable for custom marginal gain (dynamic)
    h1_per_anchor / h2_total / rel_cap: caps aligned with selector defaults
    """
    mode: str = "dynamic"
    hop_mode: str = "12"
    budget: int = 16
    budget_type: str = "edges"  # or "tokens"
    target_interval: Optional[Tuple[int, int]] = None
    cost_fn: Optional[Callable[..., int]] = None
    gain_fn: Optional[Callable[..., float]] = None
    h1_per_anchor: int = 12
    h2_total: int = 8
    rel_cap: int = 3

# ---------------------------------------------------------------------------
@dataclass
class _Maps:
    ent_name: Dict[str, str]
    rel_name: Dict[str, str]
    ent_desc: Dict[str, str]

# ---------------------------------------------------------------------------
class PromptBuilder:
    DATE_RE = re.compile(r"^(\d{4})")
    WS_RE   = re.compile(r"\s+")

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        context_mode: str = "both",   # desc | triples | both
        max_context_triples: int = 5,
        per_side_quota: Optional[Tuple[int,int]] = None,  # (S,O); None = auto 1:1
        max_name_len: int = 48,
        max_desc_len: int = 160,
        relation_cap_per_group: int = 3,
        selector_cfg: Optional[SelectorConfig] = None,     # NEW: selector options
    ) -> None:
        self.root = Path(dataset_dir)
        self.mode = context_mode
        self.k    = max_context_triples
        self.per_side_quota = per_side_quota
        self.max_name_len   = max_name_len
        self.max_desc_len   = max_desc_len
        self.rel_cap_group  = relation_cap_per_group
        self.map  = self._load_maps()
        self.selector_cfg = selector_cfg or SelectorConfig()

    # -------------------------- public -----------------------------------
    def build_prompt(self, subgraph: List[Triple], query: Triple) -> str:
        """
        Build a prompt using an explicit subgraph context (backward compatible).
        """
        s, r, o, ts, te = query
        parts: List[str] = []

        # ---- Subject / Object sections --------------------------------
        if self.mode in {"desc", "both"}:
            parts.append("### Subject")
            parts.append(f"{self._name(s)}: {self._desc(s)}")
            parts.append("\n### Object")
            parts.append(f"{self._name(o)}: {self._desc(o)}")

        # ---- Context Triples ------------------------------------------
        if self.mode in {"triples", "both"} and subgraph:
            ctx_lines = self._format_context(subgraph, anchor_s=s, anchor_o=o)
            if ctx_lines:
                parts.append("\n### Evidence (0–2 hop, most relevant)")
                parts.extend(ctx_lines)

        # ---- Question & Answer slot -----------------------------------
        rel_name = self._name(r, is_ent=False)
        parts.append("\n### Question")
        parts.append(
            "Predict the yearly time interval for the target fact.\n"
            f"- Target triple: ({self._name(s)}, {rel_name}, {self._name(o)})\n"
            "- Output strictly as 'YYYY-YYYY'. If a single year is known, repeat it (e.g., '1998-1998').\n"
            "- If unknown, use '####-####'. Provide only the final answer."
        )
        parts.append("\nANSWER: ")
        return "\n".join(parts)

    # NEW: auto-build that internally calls the selector
    def build_prompt_auto(
        self,
        query: Triple,
        *,
        selector_cfg: Optional[SelectorConfig] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a prompt by first fetching graph context via selector.
        Returns (prompt, stats) where stats summarizes selection outcomes.
        """
        cfg = selector_cfg or self.selector_cfg
        ctx, stats = self._select_context_for_query(query, cfg)
        prompt = self.build_prompt(ctx, query)
        return prompt, {"selector": stats, "k_used": len(ctx)}

    # -------------------------- mappings --------------------------------
    def _load_maps(self) -> _Maps:
        """Load index↔name/desc maps for entities and relations.
        Compatible with two styles:
          - entity2id.txt: external_id \t index [\t ...]   (use the first two cols)
          - entity2name.txt / relation2name.txt: external_id|index \t name
          - entity2desc.txt:  external_id|index \t desc
        """
        def _id2idx(kind: str) -> Dict[str, str]:
            m: Dict[str, str] = {}
            p = self.root / f"{kind}2id.txt"
            if not p.exists():
                print(f"[WARN] missing {p.name}; all names may show as <unk_...>")
                return m
            with p.open(encoding="utf-8") as f:
                for l in f:
                    if "\t" not in l:
                        continue
                    parts = l.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    wid, idx = parts[0], parts[1]
                    m[wid] = idx
            return m

        id2idx_e = _id2idx("entity")
        id2idx_r = _id2idx("relation")

        def _pretty_label(s: str) -> str:
            s = s.strip()
            s = re.sub(r'^<?https?://[^>#]*/(?:resource/|wiki/)?', '', s).strip('<>')
            s = s.replace('_', ' ')
            s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
            s = self.WS_RE.sub(" ", s).strip()
            return s

        def _name_map(kind: str, id2idx: Dict[str, str]) -> Dict[str, str]:
            m: Dict[str, str] = {}
            p = self.root / f"{kind}2name.txt"
            if not p.exists():
                return m
            with p.open(encoding="utf-8") as f:
                for l in f:
                    if "\t" not in l:
                        continue
                    left, name = l.rstrip("\n").split("\t", 1)
                    name = _pretty_label(name)
                    idx = id2idx.get(left)
                    if idx is None and left.isdigit():
                        idx = left
                    if idx is None and left in id2idx.values():
                        idx = left
                    if idx is not None:
                        m[str(idx)] = name
            return m

        ent_name = _name_map("entity",   id2idx_e)
        rel_name = _name_map("relation", id2idx_r)

        ent_desc: Dict[str, str] = {}
        desc_path = self.root / "entity2desc.txt"
        if desc_path.exists():
            with desc_path.open(encoding="utf-8") as f:
                for l in f:
                    if "\t" not in l:
                        continue
                    left, desc = l.rstrip("\n").split("\t", 1)
                    desc = _pretty_label(desc)[:512]
                    idx = id2idx_e.get(left)
                    if idx is None and left.isdigit():
                        idx = left
                    if idx is None and left in id2idx_e.values():
                        idx = left
                    if idx:
                        ent_desc[str(idx)] = desc

        if not ent_name:
            print("[WARN] entity names not loaded (check entity2name.txt format)")
        if not rel_name:
            print("[WARN] relation names not loaded (check relation2name.txt format)")
        if not ent_desc:
            print("[INFO] entity descriptions not found; fallback to [no desc]")

        return _Maps(ent_name, rel_name, ent_desc)

    # -------------------------- helpers ---------------------------------
    def _name(self, idx: EntityId, *, is_ent: bool = True) -> str:
        m = self.map.ent_name if is_ent else self.map.rel_name
        val = m.get(str(idx), f"<unk_{'ent' if is_ent else 'rel'}_{idx}>")
        return self._clip(val, self.max_name_len)

    def _desc(self, idx: EntityId) -> str:
        raw = self.map.ent_desc.get(str(idx), "[no desc]")
        raw = self.WS_RE.sub(" ", raw).strip()
        return self._clip(raw, self.max_desc_len)

    def _clip(self, text: str, n: int) -> str:
        return text if len(text) <= n else (text[: max(0, n-1)] + "…")

    def _year(self, date: str) -> Optional[str]:
        m = self.DATE_RE.match(date)
        return m.group(1) if m else None

    def _span(self, s: str, e: str) -> str:
        ys, ye = self._year(s), self._year(e)
        if ys and ye and ys == ye:
            return ys
        if ys and ye:
            return f"{ys}-{ye}"
        if ys:
            return f"since {ys}"
        if ye:
            return f"before {ye}"
        return "unknown"

    def _triple_line(self, t: Triple) -> str:
        s, r, o, ts, te = t
        return (
            f"{self._span(ts, te)}: {self._name(s)} "
            f"{self._name(r, is_ent=False)} {self._name(o)}"
        )

    # -------- context formatting (grouping/sorting/quota/dedup) -------------
    def _format_context(self, triples: List[Triple], *, anchor_s: EntityId, anchor_o: EntityId) -> List[str]:
        if not triples:
            return []

        # Deduplicate
        seen = set()
        uniq: List[Triple] = []
        for t in triples:
            key = (t[0], t[1], t[2], t[3], t[4])
            if key in seen:
                continue
            uniq.append(t); seen.add(key)

        S, O = str(anchor_s), str(anchor_o)

        def group_of(t: Triple) -> int:
            s, _, o, _, _ = t
            if (str(s) == S and str(o) == O) or (str(s) == O and str(o) == S):
                return 0  # Bridge
            if str(s) == S or str(o) == S:
                return 1  # Subject-side
            if str(s) == O or str(o) == O:
                return 2  # Object-side
            return 3      # Other

        def year_key(t: Triple):
            ys = self._year(t[3]); ye = self._year(t[4])
            if ys is None and ye is None: return (10**9, 10**9)
            if ys is None: return (10**9, int(ye))
            if ye is None: return (int(ys), 10**9)
            return (int(ys), int(ye))

        grouped = {0:[], 1:[], 2:[], 3:[]}
        for t in uniq:
            grouped[group_of(t)].append(t)
        for gid in grouped:
            grouped[gid].sort(key=year_key)

        # Per-group relation diversity
        def take_with_rel_cap(ts: List[Triple], cap_per_rel: int) -> List[Triple]:
            from collections import Counter
            out, c = [], Counter()
            for t in ts:
                r = str(t[1])
                if c[r] >= cap_per_rel:
                    continue
                out.append(t); c[r] += 1
            return out

        grouped[1] = take_with_rel_cap(grouped[1], self.rel_cap_group)
        grouped[2] = take_with_rel_cap(grouped[2], self.rel_cap_group)

        # Bridge first, then split remaining quota evenly between S/O
        k_total = max(0, self.k)
        bridge = grouped[0][:min(len(grouped[0]), k_total)]
        remain = k_total - len(bridge)

        if self.per_side_quota is None:
            q_s = remain // 2
            q_o = remain - q_s
        else:
            q_s, q_o = self.per_side_quota
            if q_s + q_o > remain:
                scale = remain / max(1, q_s + q_o)
                q_s = int(q_s * scale); q_o = remain - q_s

        pick_s = grouped[1][:q_s]
        pick_o = grouped[2][:q_o]
        used   = len(bridge) + len(pick_s) + len(pick_o)

        extra = []
        if used < k_total:
            extra = grouped[3][: (k_total - used)]

        selected = bridge + pick_s + pick_o + extra  # noqa: F841

        lines: List[str] = []
        if bridge:
            lines.append("- [Bridge]")
            lines.extend("  - " + self._triple_line(t) for t in bridge)
        if pick_s:
            lines.append("- [Subject-side]")
            lines.extend("  - " + self._triple_line(t) for t in pick_s)
        if pick_o:
            lines.append("- [Object-side]")
            lines.extend("  - " + self._triple_line(t) for t in pick_o)
        if extra:
            lines.append("- [Other]")
            lines.extend("  - " + self._triple_line(t) for t in extra)

        return lines

    # ---------------------- selector bridge (lazy import) --------------------
    def _select_context_for_query(
        self,
        query: Triple,
        cfg: SelectorConfig,
    ) -> Tuple[List[Triple], Dict[str, Any]]:
        """
        Fetch graph context for (s,o) using either dynamic or static selector.
        Lazy-imports selector to avoid circular imports at module load time.
        """
        s, _, o, _, _ = query

        # Late import avoids circular dependency with selector.py
        from . import selector as _sel  # noqa: WPS433

        if cfg.mode == "none":
            return [], {"mode": "none", "budget_used": 0}

        if cfg.mode == "dynamic":
            triples, stats = _sel.select_context_dynamic(
                kg_root=self.root,
                s=int(s), o=int(o),
                hop_mode=cfg.hop_mode,
                budget=cfg.budget,
                budget_type=cfg.budget_type,
                target_interval=cfg.target_interval,
                cost_fn=cfg.cost_fn or _sel._default_cost_fn,     # fall back to default
                gain_fn=cfg.gain_fn or _sel._default_gain_fn,     # fall back to default
                h1_per_anchor=cfg.h1_per_anchor,
                h2_total=cfg.h2_total,
                rel_cap=cfg.rel_cap,
            )
            return triples, {"mode": "dynamic", **stats}

        # cfg.mode == "static"
        triples, n1, n2 = _sel.select_context(
            kg_root=self.root,
            s=int(s), o=int(o),
            hop_mode=cfg.hop_mode,
            h1_per_anchor=cfg.h1_per_anchor,
            h2_total=cfg.h2_total,
        )
        return triples, {"mode": "static", "n1": n1, "n2": n2, "k": len(triples)}

# ---------------- quick test --------------------
if __name__ == "__main__":
    import json, argparse
    ap = argparse.ArgumentParser("Structured PromptBuilder test (auto+static)")
    ap.add_argument("root")
    ap.add_argument("triple")
    ap.add_argument("--ctx", default="[]")
    ap.add_argument("--mode", default="both", choices=["desc","triples","both"])
    ap.add_argument("--auto", action="store_true", help="use selector to fetch context")
    ap.add_argument("--selector", default="dynamic", choices=["dynamic","static","none"])
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--budget_type", default="edges", choices=["edges","tokens"])
    ap.add_argument("--hop_mode", default="12", choices=["0","1","12"])
    a = ap.parse_args()

    pb = PromptBuilder(
        a.root,
        context_mode=a.mode,
        selector_cfg=SelectorConfig(
            mode=a.selector,
            hop_mode=a.hop_mode,
            budget=a.budget,
            budget_type=a.budget_type,
        ),
    )

    q   = tuple(json.loads(a.triple))
    if a.auto:
        prompt, stats = pb.build_prompt_auto(q)
        print("\n=== PROMPT (AUTO) ===\n")
        print(prompt)
        print("\n=== SELECTOR STATS ===\n")
        print(json.dumps(stats, indent=2))
    else:
        ctx = [tuple(x) for x in json.loads(a.ctx)]
        print("\n=== PROMPT (MANUAL CTX) ===\n")
        print(pb.build_prompt(ctx, q))
