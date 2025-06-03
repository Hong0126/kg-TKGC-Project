import torch

def gIOU(pred, gold):
    s1,e1 = pred[:,0], pred[:,1]
    s2,e2 = gold[:,0], gold[:,1]
    inter = torch.clamp_min(torch.min(e1,e2)-torch.max(s1,s2), 0)
    union = torch.max(e1,e2) - torch.min(s1,s2)
    return (inter / union.clamp(min=1e-6)).mean().item()