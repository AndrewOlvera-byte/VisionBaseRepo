from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from utils.logger import get_logger
from utils.metrics import accuracy
from utils.checkpoint import save_checkpoint

logger = get_logger(__name__)

def run_train(cfg: DictConfig):
    """Single-GPU training loop.
    Args:
        cfg: Hydra DictConfig coming from configs/train.yaml
    """
    # 1. ==== data ====
    dm = hydra.utils.instantiate(cfg.dataset)
    train_loader, val_loader = dm.train_dataloader(), dm.val_dataloader()

    # 2. ==== model ====
    model = hydra.utils.instantiate(cfg.model).cuda()
    try:
        model = torch.compile(model)  # PyTorch ≥2.0
    except Exception:
        logger.warning("torch.compile failed – continuing without compilation")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_cls = getattr(torch.optim, cfg.optimizer.name.capitalize()) if isinstance(cfg.optimizer.name, str) else torch.optim.AdamW
    optimizer = optimizer_cls(model.parameters(), lr=cfg.optimizer.lr)

    scheduler = None
    if "scheduler" in cfg and cfg.scheduler is not None:
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_acc = 0.0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    # 3. ==== loop ====
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                preds = model(xb)
                loss = criterion(preds, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * xb.size(0)

        if scheduler is not None:
            scheduler.step()

        # === validation ===
        val_acc = _evaluate(model, val_loader, cfg.amp)
        logger.info(f"Epoch {epoch:03d} | loss={epoch_loss/len(train_loader.dataset):.4f} | val_acc={val_acc*100:.2f}%")

        # === checkpoint ===
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, ckpt_dir / "best.pt", epoch, best_acc)

        # Always save last epoch
        save_checkpoint(model, optimizer, ckpt_dir / "last.pt", epoch, val_acc)

        # === Ray Tune metric ===
        try:
            from ray import tune  # noqa: WPS433  (runtime import)
            tune.report(val_accuracy=val_acc, epoch=epoch, loss=epoch_loss/len(train_loader.dataset))
        except ImportError:
            pass


def _evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, amp: bool) -> float:
    model.eval()
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