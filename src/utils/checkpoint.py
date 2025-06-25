from pathlib import Path
import torch


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path, epoch: int, metric: float):
    """Save model checkpoint to *path* including optimizer state."""
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metric": metric,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path) 