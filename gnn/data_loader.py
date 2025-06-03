# data_loader.py  —— 单张大图 + 年份归一化
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import re, torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

# 年份反/归一化区间
Y_MIN, Y_MAX = 0, 2025
_mid, _half = (Y_MIN + Y_MAX) / 2, (Y_MAX - Y_MIN) / 2
def _norm(y: int) -> float:
    return -1. if y == -1 else (y - _mid) / _half

_YEAR = re.compile(r"\d{4}")
def _parse_year(tok: str) -> int:
    m = _YEAR.match(tok); return int(m.group()) if m else -1

class TKGC_Dataset(Dataset):
    def __init__(self, root: str | Path, split: str):
        root = Path(root)
        path = root / f"{split.lower()}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        self.graph = self._build(path)

    def __len__(self):  return 1
    def __getitem__(self, idx): return self.graph

    @staticmethod
    def _build(path: Path) -> Data:
        ents, rels, rows = {}, {}, []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: parts += ["####-##-##"] * (5 - len(parts))
                s, r, o, ts, te = parts[:5]
                ents.setdefault(s, len(ents)); ents.setdefault(o, len(ents))
                rels.setdefault(r, len(rels))
                ys, ye = _norm(_parse_year(ts)), _norm(_parse_year(te))
                rows.append((ents[s], rels[r], ents[o], ys, ye))

        rows_t = torch.tensor(rows)
        src, rel, dst = rows_t[:,0].long(), rows_t[:,1].long(), rows_t[:,2].long()   # **修改**
        ys, ye        = rows_t[:,3].float(), rows_t[:,4].float()                     # **修改**

        data = Data()
        data.edge_index = torch.stack([src, dst]).long()   # **修改**

        data.edge_type  = rel
        data.edge_t     = torch.stack([ys, ye], 1).float()
        data.num_nodes  = len(ents)
        print(f"[{path.name}] 节点 {data.num_nodes}, 边 {data.edge_index.size(1)}")
        return data

