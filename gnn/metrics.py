import torch

def calculate_iou(pred, true):
    # pred and true are tensors of shape [B, 2]
    pred_start, pred_end = pred[:, 0], pred[:, 1]
    true_start, true_end = true[:, 0], true[:, 1]
    
    intersection_start = torch.max(pred_start, true_start)
    intersection_end = torch.min(pred_end, true_end)
    
    intersection_len = torch.clamp(intersection_end - intersection_start, min=0)
    
    union_len = (pred_end - pred_start) + (true_end - true_start) - intersection_len
    
    return intersection_len / (union_len + 1e-9)

def calculate_giou(pred, true):
    iou = calculate_iou(pred, true)
    
    c_start = torch.min(pred[:, 0], true[:, 0])
    c_end = torch.max(pred[:, 1], true[:, 1])
    c_len = torch.clamp(c_end - c_start, min=0)
    
    union_len = (pred[:, 1] - pred[:, 0]) + (true[:, 1] - true[:, 0]) - (iou * c_len)
    
    return iou - (c_len - union_len) / (c_len + 1e-9)

def calculate_diou_loss(pred, true):
    iou = calculate_iou(pred, true)
    
    pred_center = (pred[:, 0] + pred[:, 1]) / 2
    true_center = (true[:, 0] + true[:, 1]) / 2
    
    c_start = torch.min(pred[:, 0], true[:, 0])
    c_end = torch.max(pred[:, 1], true[:, 1])
    c_len_sq = (c_end - c_start) ** 2
    
    center_dist_sq = (pred_center - true_center) ** 2
    
    diou = iou - (center_dist_sq / (c_len_sq + 1e-9))
    
    return 1 - diou