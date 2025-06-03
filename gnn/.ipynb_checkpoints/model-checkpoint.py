# model.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from time_embed import sinusoid
from config import CFG

class TimeAwareGAT(nn.Module):
    def __init__(self, num_nodes: int, num_rel: int):
        super().__init__()
        hid, heads = CFG.num_hid, CFG.num_heads
        self.ent = nn.Embedding(num_nodes, hid)
        self.rel = nn.Embedding(num_rel,   hid)
        self.time_proj = nn.Linear(CFG.time_dim * 2, hid)

        self.gat1 = GATConv(hid, hid // heads, heads=heads, edge_dim=hid, dropout=CFG.dropout)
        self.gat2 = GATConv(hid, hid // heads, heads=heads, edge_dim=hid, dropout=CFG.dropout)
        self.out  = nn.Linear(hid, 2)   # (center_raw, span_raw)

    # -------- 整图节点向量 --------
    def _node_repr(self, data):
        x = self.ent.weight
        e_attr = self.rel(data.edge_type) + self._time_vec(data)
        h1 = self.gat1(x, data.edge_index, edge_attr=e_attr)
        h2 = self.gat2(h1, data.edge_index, edge_attr=e_attr)
        return h1 + h2                  # 残差

    # -------- 时间向量 --------
    def _time_vec(self, data):
        y_s = data.edge_t[:, 0:1]; y_e = data.edge_t[:, 1:2]
        ts, te = sinusoid(y_s, CFG.time_dim), sinusoid(y_e, CFG.time_dim)
        return self.time_proj(torch.cat([ts, te], 1))

    # -------- 边级区间预测 (归一化空间) --------
    def forward_on_edges(self, data, eids):
        h = self._node_repr(data)
        src, dst = data.edge_index[0, eids], data.edge_index[1, eids]
        h_e = (h[src] + h[dst]) / 2

        center_raw, span_raw = self.out(h_e).split(1, dim=-1)
        center_norm = torch.tanh(center_raw)                    # (-1,1)
        span_norm   = (CFG.span_max / CFG.HALF) * torch.tanh(span_raw)  # 再换算到归一化
        t_start = center_norm - span_norm / 2
        t_end   = center_norm + span_norm / 2
        pred_norm = torch.cat([t_start, t_end], 1)
        return torch.sort(pred_norm, dim=-1)[0]                 # 保证顺序
