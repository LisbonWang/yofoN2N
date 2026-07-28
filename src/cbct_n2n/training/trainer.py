from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import save_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _source_target(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        source = batch["source"] if "source" in batch else batch["even"]
        return source, batch["target"]
    return batch[0], batch[1]


def train_one_epoch(model, loader, optimizer, device: torch.device, scaler=None, gradient_clip: float | None = None) -> float:
    model.train()
    total, count = 0.0, 0
    amp_enabled = scaler is not None and device.type == "cuda"
    for batch in loader:
        source, target = (item.to(device, non_blocking=True) for item in _source_target(batch))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            loss = F.l1_loss(model(source), target)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        batch_size = source.shape[0]
        total += float(loss.detach()) * batch_size
        count += batch_size
    if not count:
        raise ValueError("Training loader is empty")
    return total / count


@torch.no_grad()
def validate(model, loader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        source, target = (item.to(device, non_blocking=True) for item in _source_target(batch))
        loss = F.l1_loss(model(source), target)
        total += float(loss) * source.shape[0]
        count += source.shape[0]
    if not count:
        raise ValueError("Validation loader is empty")
    return total / count


def fit(model, train_loader, val_loader, optimizer, scheduler, device: torch.device, run_dir: str | Path, config: dict[str, Any], *, start_epoch: int = 0) -> dict[str, float]:
    options = config["training"]
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(options.get("amp", True) and device.type == "cuda"))
    best_loss, stale = float("inf"), 0
    best_path = destination / "best.pt"
    if start_epoch > 0 and best_path.exists():
        historical = torch.load(best_path, map_location="cpu", weights_only=False)
        best_loss = float(historical.get("metrics", {}).get("val_loss", best_loss))
    best_metrics = {"val_loss": best_loss}
    history_path = destination / "history.jsonl"
    for epoch in range(start_epoch, int(options["epochs"])):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, options.get("gradient_clip"))
        val_loss = validate(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"]}
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        save_checkpoint(destination / "last.pt", model, optimizer, scheduler, epoch, metrics, config)
        if val_loss < best_loss:
            best_loss, stale, best_metrics = val_loss, 0, metrics
            save_checkpoint(destination / "best.pt", model, optimizer, scheduler, epoch, metrics, config)
            status = " best"
        else:
            stale += 1
            status = ""
        print(
            f"epoch={epoch + 1}/{options['epochs']} train_l1={train_loss:.6f} "
            f"val_l1={val_loss:.6f} lr={metrics['learning_rate']:.3e}{status}",
            flush=True,
        )
        if status == "":
            if stale >= int(options.get("patience", options["epochs"])):
                break
    return best_metrics
