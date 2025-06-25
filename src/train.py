from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from utils.logger import get_logger
from utils.checkpoint import save_checkpoint

logger = get_logger(__name__)

def run_train(cfg: DictConfig):
    """Single-GPU training loop with optional optimizations."""
    # === global optimisations ===
    torch.backends.cudnn.benchmark = True  # enable cudnn autotuner

    grad_acc_steps = int(cfg.get("gradient_accumulation_steps", 1))

    # 1. ==== data ====
    dm = hydra.utils.instantiate(cfg.dataset)
    from utils.prefetch import PrefetchLoader  # local import avoids circular
    train_loader = PrefetchLoader(dm.train_dataloader())
    val_loader = PrefetchLoader(dm.val_dataloader())
    num_train_samples = len(dm.train_dataloader().dataset)
    steps_per_epoch = len(dm.train_dataloader())

    # 2. ==== model ====
    model = hydra.utils.instantiate(cfg.model).cuda()

    if bool(cfg.get("compile", False)):
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"torch.compile failed: {exc}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_cls = getattr(torch.optim, cfg.optimizer.name, None)
    if optimizer_cls is None:
        # fallback to capitalised CamelCase
        camel = cfg.optimizer.name.title().replace("_", "")
        optimizer_cls = getattr(torch.optim, camel, torch.optim.AdamW)
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
        optimizer.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(train_loader, start=1):
            # PrefetchLoader already moved tensors to GPU (non_blocking).
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                preds = model(xb)
                loss = criterion(preds, yb) / grad_acc_steps

            scaler.scale(loss).backward()

            if step % grad_acc_steps == 0 or step == steps_per_epoch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * xb.size(0) * grad_acc_steps

        if scheduler is not None:
            scheduler.step()

        # === validation ===
        val_acc = _evaluate(model, val_loader, cfg.amp)
        logger.info(f"Epoch {epoch:03d} | loss={epoch_loss/num_train_samples:.4f} | val_acc={val_acc*100:.2f}%")

        # === checkpoint ===
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, ckpt_dir / "best.pt", epoch, best_acc)

        # Always save last epoch
        save_checkpoint(model, optimizer, ckpt_dir / "last.pt", epoch, val_acc)

        # === Ray Tune metric ===
        try:
            from ray import tune  # noqa: WPS433  (runtime import)
            tune.report(val_accuracy=val_acc, epoch=epoch, loss=epoch_loss/num_train_samples)
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