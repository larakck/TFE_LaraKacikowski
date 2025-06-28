import torch

def dice_score(preds, targets):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum((1,2,3))
    union = preds.sum((1,2,3)) + targets.sum((1,2,3))
    return ((2. * intersection + 1e-7) / (union + 1e-7)).mean().item()
