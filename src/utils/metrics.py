import torch
try:
    from torchmetrics.functional import accuracy as _tm_accuracy
except ImportError:  # fallback if torchmetrics absent
    _tm_accuracy = None


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return classification accuracy as a scalar tensor."""
    if _tm_accuracy is not None:
        return _tm_accuracy(preds, targets, task="multiclass", num_classes=preds.shape[1])
    else:
        pred_labels = torch.argmax(preds, dim=1)
        return (pred_labels == targets).float().mean() 