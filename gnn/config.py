from dataclasses import dataclass
from pathlib import Path

@dataclass
class CFG:
    # ----- 文件路径 -----
    data_dir   : Path = Path("/root/tkgc_data/share/WIKIDATA12k/")   # 目录里需有 train.txt/valid.txt/test.txt
    out_dir    : Path = Path("gnn_ckpt")

    Y_MIN: int = 0
    Y_MAX: int = 2025
    HALF:  float = (Y_MAX - Y_MIN) / 2  # 方便模型引用

    # ---------- 模型 ----------
    num_hid   = 512
    num_heads = 4
    time_dim  = 64
    dropout   = 0.2
    span_max  = 200.0            # 年份绝对跨度上限

    # ---------- 训练 ----------
    lr        = 5e-4
    batches   = 1024
    epochs    = 100
    seed      = 42
