from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig

from utils.logger import get_logger
from utils.metrics import accuracy

logger = get_logger(__name__)


def run_eval(cfg: DictConfig):
    """Evaluate a checkpoint on the validation (and optionally test) set."""
    ckpt_path = Path(cfg.get("eval_checkpoint", "checkpoints/best.pt"))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Pass eval_checkpoint=<path>.")

    # Data
    dm = hydra.utils.instantiate(cfg.dataset)
    val_loader = dm.val_dataloader()

    # Model
    model = hydra.utils.instantiate(cfg.model).cuda()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    # Accuracy
    acc = _evaluate(model, val_loader, cfg.amp)
    logger.info(f"Validation accuracy: {acc*100:.2f}%")

    if cfg.get("run_test_set", False):
        test_loader = dm.test_dataloader() if hasattr(dm, "test_dataloader") else None
        if test_loader is not None:
            acc_test = _evaluate(model, test_loader, cfg.amp)
            logger.info(f"Test accuracy: {acc_test*100:.2f}%")
        else:
            logger.warning("DataModule has no test_dataloader() method; skipping test evaluation.")


def _evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, amp: bool) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / max(1, total) 