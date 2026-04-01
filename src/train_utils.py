from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

__all__ = [
    "AverageMeter",
    "EpochMeters",
    "append_jsonl",
    "count_parameters",
    "ensure_dir",
    "format_seconds",
    "load_checkpoint",
    "save_checkpoint",
    "save_json",
    "set_seed",
    "timestamp_run_name",
]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_run_name(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")


def set_seed(seed: int | None) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def format_seconds(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


@dataclass
class EpochMeters:
    loc: AverageMeter = field(default_factory=AverageMeter)
    conf: AverageMeter = field(default_factory=AverageMeter)
    landm: AverageMeter = field(default_factory=AverageMeter)
    total: AverageMeter = field(default_factory=AverageMeter)

    def update(self, loss_loc: float, loss_conf: float, loss_landm: float, total: float) -> None:
        self.loc.update(loss_loc)
        self.conf.update(loss_conf)
        self.landm.update(loss_landm)
        self.total.update(total)

    def as_dict(self) -> dict[str, float]:
        return {
            "loss_loc": self.loc.avg,
            "loss_conf": self.conf.avg,
            "loss_landm": self.landm.avg,
            "loss_total": self.total.avg,
        }


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    best_loss: float,
    config: dict[str, Any],
    args: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)

    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_loss": float(best_loss),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "args": args,
        "metrics": metrics or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device)
    model.load_state_dict(checkpoint["model"], strict=strict)

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint
