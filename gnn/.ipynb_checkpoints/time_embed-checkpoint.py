# time_embed.py
import math, torch

def sinusoid(year: torch.Tensor, dim: int = 64) -> torch.Tensor:
    year = year.float()
    device = year.device                              # ==== 修改点 ====
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float, device=device)
        * -(math.log(10000.0) / dim)
    )
    sin = torch.sin(year * div)
    cos = torch.cos(year * div)
    out = torch.zeros(year.size(0), dim, device=device)
    out[:,0::2], out[:,1::2] = sin, cos
    return out
