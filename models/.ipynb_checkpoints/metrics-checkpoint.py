def gIOU(pred, gold):
    """pred, gold = (start, end)."""
    a1, a2 = pred
    b1, b2 = gold
    inter = max(0, min(a2, b2) - max(a1, b1))
    union = max(a2, b2) - min(a1, b1)
    return inter / union if union else 0.0
# TODO: aeIOU, gaeIOU, Hits@k, MRR …
