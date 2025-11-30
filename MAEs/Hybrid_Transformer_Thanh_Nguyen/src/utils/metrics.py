import torch
from torch import Tensor


# Functions for BCEWithLogitsLoss accuracy metrics
def accuracy_metric_bce(outputs: Tensor, target: Tensor) -> float:
    """
    Computes accuracy for binary classification when using BCEWithLogitsLoss.
    Applies sigmoid on raw logits, then thresholds at 0.5.
    """
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float().view(-1)
    target = target.view(-1)
    
    return torch.sum(preds == target).item() / target.size(0)


# Functions for CrossEntropyLoss accuracy metrics
def accuracy_metric_ce(pred: Tensor, target: Tensor) -> float:
    r"""
    Accuracy for multi-class classification (using CrossEntropyLoss).
    """
    pred_class = pred.argmax(dim=1)

    # Check if target is one-hot encoded
    if target.dim() > 1 and target.shape[1] > 1:
        target_class = target.argmax(dim=1)  # convert one-hot to class indices
    else:
        target_class = target  # already class indices

    return torch.sum(pred_class == target_class).item() / target.size(0)