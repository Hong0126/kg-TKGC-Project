# train_gnn.py
import random, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm

from config import CFG
from data_loader import TKGC_Dataset
from model import TimeAwareGAT

# 反归一化
def denorm(y):
    return y * CFG.HALF + (CFG.Y_MIN + CFG.Y_MAX) / 2

def giou(pred_norm, gold_norm):
    p, g = denorm(pred_norm), denorm(gold_norm)
    s1,e1 = p[:,0], p[:,1]
    s2,e2 = g[:,0], g[:,1]
    inter = (torch.min(e1,e2) - torch.max(s1,s2)).clamp(min=0)
    union = torch.max(e1,e2) - torch.min(s1,s2)
    return (inter / union.clamp(min=1)).mean()

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    set_seed(CFG.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = Path("/root/tkgc_data/share/WIKIDATA12k")  # 数据目录
    g_tr = TKGC_Dataset(root, "train")[0].to(dev)
    g_va = TKGC_Dataset(root, "valid")[0].to(dev)

    tr_ids = torch.nonzero((g_tr.edge_t != -1).all(1), as_tuple=False).view(-1)
    va_ids = torch.nonzero((g_va.edge_t != -1).all(1), as_tuple=False).view(-1)

    model = TimeAwareGAT(g_tr.num_nodes, int(g_tr.edge_type.max())+1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.epochs)

    get_loader = lambda ids: DataLoader(ids, batch_size=CFG.batches, shuffle=True)

    for ep in range(1, CFG.epochs+1):
        # ----- train -----
        model.train(); tot=0
        for eids in tqdm(get_loader(tr_ids), desc=f"Ep{ep}", leave=False):
            eids = eids.to(dev)
            gold = g_tr.edge_t[eids]
            pred = model.forward_on_edges(g_tr, eids)

            l1 = F.smooth_l1_loss(pred, gold)
            giou_loss = 1 - giou(pred, gold)
            loss = l1 + 0.5 * giou_loss
            loss.backward(); opt.step(); opt.zero_grad()
            tot += loss.item()
        sched.step()

        # ----- valid -----
        # ---------- 验证 ----------
        model.eval()
        with torch.no_grad():
            pv = model.forward_on_edges(g_va, va_ids.to(dev))
            gv = g_va.edge_t[va_ids]
        
            # === 1) 统计跨度 ===
            span = denorm(pv[:,1]) - denorm(pv[:,0])
            print(f"[DEBUG] span avg={span.mean():.1f}y   max={span.max():.1f}y")
        
            # === 2) 中心偏移 ===
            center_pred = (denorm(pv[:,0]) + denorm(pv[:,1])) / 2
            center_gold = (denorm(gv[:,0]) + denorm(gv[:,1])) / 2
            print(f"[DEBUG] center_pred avg={center_pred.mean():.1f}, "
                  f"center_gold avg={center_gold.mean():.1f}, "
                  f"diff={abs(center_pred-center_gold).mean():.1f}")
        
            # === 3) 打印 5 条样本 ===
            idx = torch.randint(0, pv.size(0), (5,))
            print("[DEBUG] sample pred|gold:")
            for i in idx:
                print("  ", denorm(pv[i]).cpu().numpy(), "|", denorm(gv[i]).cpu().numpy())
        
            # 原有 gIOU 计算
            g_val = giou(pv, gv).item()


        print(f"Ep{ep:02d} | loss {tot/len(get_loader(tr_ids)):.3f} | gIOU {g_val:.4f} | lr {sched.get_last_lr()[0]:.2e}")

if __name__ == "__main__":
    main()
