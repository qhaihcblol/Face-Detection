from __future__ import annotations

import argparse
import itertools
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config.config import get_config
from src.data.transforms import RetinaFaceTrainTransform
from src.data.widerface import WiderFaceDataset, widerface_collate
from src.detection.prior_box import PriorBox
from src.losses.multibox_loss import MultiBoxLoss
from src.models.retinaface import RetinaFace
from src.train_utils import (
    EpochMeters,
    append_jsonl,
    count_parameters,
    ensure_dir,
    format_seconds,
    load_checkpoint,
    save_checkpoint,
    save_json,
    set_seed,
    timestamp_run_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RetinaFace on WIDER FACE train split"
    )
    parser.add_argument("--config", default="mobilenet_v2", help="config/backbone name")
    parser.add_argument(
        "--train-root", default="datasets/train", help="train split root"
    )
    parser.add_argument(
        "--annotation-file", default=None, help="override annotation file path"
    )
    parser.add_argument("--image-root", default=None, help="override image root path")
    parser.add_argument(
        "--output-dir", default="outputs/train", help="directory for training runs"
    )
    parser.add_argument("--run-name", default=None, help="custom run directory name")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device, e.g. cuda, cuda:0, cpu",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of dataloader workers"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="override config batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="override config epochs"
    )
    parser.add_argument(
        "--image-size", type=int, default=None, help="override config image size"
    )
    parser.add_argument(
        "--min-face-size",
        type=float,
        default=0.0,
        help="minimum resized face size kept during random crop augmentation",
    )
    parser.add_argument(
        "--overlap-thresh",
        type=float,
        default=0.30,
        help="IoU threshold for positive anchor matching",
    )
    parser.add_argument(
        "--best-prior-iou-floor",
        type=float,
        default=0.10,
        help="minimum best-prior IoU required before force matching a GT",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="base learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR scheduler gamma")
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1000,
        help="linear warmup iterations before using base LR",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="save periodic checkpoints every N epochs",
    )
    parser.add_argument(
        "--log-interval", type=int, default=20, help="log every N optimizer steps"
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=0.0,
        help="clip gradient norm if > 0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for Python, NumPy, and PyTorch",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable automatic mixed precision on CUDA",
    )
    parser.add_argument(
        "--pretrain",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override backbone pretrained weights usage",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="for debugging: stop each epoch after this many batches (0 = full epoch)",
    )
    parser.add_argument(
        "--tqdm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show tqdm progress bar during training",
    )
    return parser.parse_args()


def build_run_dir(args: argparse.Namespace) -> Path:
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if resume_path.parent.name == "checkpoints":
            return resume_path.parent.parent
        return resume_path.parent

    run_name = args.run_name or timestamp_run_name(prefix=f"retinaface_{args.config}")
    return ensure_dir(Path(args.output_dir) / run_name)


def build_dataset(args: argparse.Namespace, cfg: dict[str, Any]) -> WiderFaceDataset:
    train_root = Path(args.train_root)
    annotation_file = (
        Path(args.annotation_file)
        if args.annotation_file
        else train_root / "train_annotations.txt"
    )
    image_root = Path(args.image_root) if args.image_root else train_root / "images"

    transform = RetinaFaceTrainTransform(
        image_size=int(cfg["image_size"]),
        min_face_size=float(args.min_face_size),
    )
    return WiderFaceDataset(
        annotation_file=annotation_file,
        image_root=image_root,
        transform=transform,
        normalize_targets=True,
        to_tensor=True,
    )


def build_dataloader(
    dataset: WiderFaceDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=widerface_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def move_images_to_device(
    images: Tensor | Sequence[Tensor], device: torch.device
) -> Tensor:
    if isinstance(images, Tensor):
        return images.to(device=device, non_blocking=device.type == "cuda")

    if isinstance(images, Sequence) and all(
        isinstance(image, Tensor) for image in images
    ):
        image_shapes = {tuple(image.shape) for image in images}
        if len(image_shapes) != 1:
            raise ValueError(
                "images must have the same shape to be stacked for training"
            )
        batch = torch.stack(list(images), dim=0)
        return batch.to(device=device, non_blocking=device.type == "cuda")

    raise TypeError("batch images must be a Tensor or a sequence of Tensors")


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_lrs: list[float],
    scale: float,
) -> None:
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale


def build_grad_scaler(enabled: bool):
    return torch.GradScaler(enabled=enabled)


def train_one_epoch(
    *,
    model: RetinaFace,
    criterion: MultiBoxLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    data_loader: DataLoader,
    priors: Tensor,
    device: torch.device,
    epoch: int,
    global_step: int,
    warmup_iters: int,
    base_lrs: list[float],
    log_interval: int,
    clip_grad_norm: float,
    limit_batches: int,
    use_amp: bool,
    show_progress: bool,
    log: Callable[..., None],
) -> tuple[EpochMeters, int, float]:
    model.train()
    criterion.train()

    meters = EpochMeters()
    start_time = time.time()
    num_batches = len(data_loader)
    effective_total = (
        min(num_batches, limit_batches) if limit_batches > 0 else num_batches
    )

    data_iterable = (
        itertools.islice(data_loader, limit_batches)
        if limit_batches > 0
        else data_loader
    )
    progress = tqdm(
        total=effective_total,
        desc=f"Epoch {epoch + 1}",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )

    try:
        for batch_idx, (images, targets) in enumerate(data_iterable):

            if warmup_iters > 0 and global_step < warmup_iters:
                warmup_scale = float(global_step + 1) / float(warmup_iters)
                set_learning_rate(optimizer, base_lrs, warmup_scale)

            images = move_images_to_device(images, device)
            optimizer.zero_grad(set_to_none=True)

            autocast_context = (
                torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=True
                )
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                predictions = model(images)
                loss_loc, loss_conf, loss_landm = criterion(
                    predictions, priors, targets
                )
                loss_total = criterion.combine_losses(loss_loc, loss_conf, loss_landm)

            scaler.scale(loss_total).backward()
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            meters.update(
                loss_loc=float(loss_loc.detach()),
                loss_conf=float(loss_conf.detach()),
                loss_landm=float(loss_landm.detach()),
                total=float(loss_total.detach()),
            )
            lr = optimizer.param_groups[0]["lr"]
            progress.update(1)
            progress.set_postfix(
                lr=f"{lr:.2e}",
                loss=f"{meters.total.avg:.4f}",
                loc=f"{meters.loc.avg:.4f}",
                conf=f"{meters.conf.avg:.4f}",
                landm=f"{meters.landm.avg:.4f}",
                refresh=batch_idx + 1 == effective_total,
            )

            if (
                batch_idx % max(log_interval, 1) == 0
                or batch_idx + 1 == effective_total
            ):
                elapsed = time.time() - start_time
                processed_batches = batch_idx + 1
                eta_seconds = (
                    elapsed
                    / max(processed_batches, 1)
                    * max(
                        effective_total - processed_batches,
                        0,
                    )
                )
                log(
                    f"epoch {epoch + 1} step {processed_batches}/{effective_total} "
                    f"lr={lr:.6f} "
                    f"loss={meters.total.avg:.4f} "
                    f"(loc={meters.loc.avg:.4f}, conf={meters.conf.avg:.4f}, landm={meters.landm.avg:.4f}) "
                    f"eta={format_seconds(eta_seconds)}",
                    echo=not show_progress,
                )
    finally:
        progress.close()

    epoch_time = time.time() - start_time
    return meters, global_step, epoch_time


def main() -> None:
    args = parse_args()
    run_dir = build_run_dir(args)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    metrics_path = run_dir / "metrics.jsonl"
    log_path = run_dir / "train.log"

    def log(message: str, *, echo: bool = True) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        if echo:
            if args.tqdm:
                tqdm.write(line)
            else:
                print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    set_seed(args.seed)
    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    cfg = get_config(args.config)
    if args.batch_size is None:
        args.batch_size = int(cfg["batch_size"])
    if args.epochs is None:
        args.epochs = int(cfg["epochs"])
    if args.image_size is not None:
        cfg["image_size"] = int(args.image_size)
    if args.pretrain is not None:
        cfg["pretrain"] = bool(args.pretrain)

    dataset = build_dataset(args, cfg)
    data_loader = build_dataloader(dataset, args, device)

    model = RetinaFace(cfg).to(device)
    criterion = MultiBoxLoss(
        cfg,
        overlap_thresh=float(args.overlap_thresh),
        best_prior_iou_floor=float(args.best_prior_iou_floor),
    ).to(device)
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[int(milestone) for milestone in cfg["milestones"]],
        gamma=args.gamma,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = build_grad_scaler(use_amp)

    base_lrs = [group["lr"] for group in optimizer.param_groups]
    priors = (
        PriorBox(
            cfg,
            image_size=(int(cfg["image_size"]), int(cfg["image_size"])),
        )
        .forward()
        .to(device)
    )

    start_epoch = 0
    global_step = 0
    best_loss = math.inf
    current_epoch = start_epoch - 1

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_loss = float(checkpoint.get("best_loss", math.inf))
        log(
            f"resumed from {args.resume} at epoch {start_epoch + 1} "
            f"(global_step={global_step}, best_loss={best_loss:.4f})"
        )

    save_json(run_dir / "config.json", cfg)
    save_json(run_dir / "args.json", vars(args))
    save_json(
        run_dir / "run_info.json",
        {
            "device": str(device),
            "dataset_size": len(dataset),
            "trainable_parameters": count_parameters(model),
            "num_priors": int(priors.size(0)),
            "use_amp": use_amp,
        },
    )

    log(
        f"starting training on {device} with {len(dataset)} samples, "
        f"batch_size={args.batch_size}, epochs={args.epochs}, "
        f"trainable_params={count_parameters(model):,}"
    )

    try:
        for epoch in range(start_epoch, args.epochs):
            current_epoch = epoch
            epoch_meters, global_step, epoch_time = train_one_epoch(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                data_loader=data_loader,
                priors=priors,
                device=device,
                epoch=epoch,
                global_step=global_step,
                warmup_iters=args.warmup_iters,
                base_lrs=base_lrs,
                log_interval=args.log_interval,
                clip_grad_norm=args.clip_grad_norm,
                limit_batches=args.limit_batches,
                use_amp=use_amp,
                show_progress=args.tqdm,
                log=log,
            )

            if global_step >= args.warmup_iters:
                scheduler.step()

            summary = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_sec": epoch_time,
                **epoch_meters.as_dict(),
            }
            append_jsonl(metrics_path, summary)
            save_json(run_dir / "latest_metrics.json", summary)

            if summary["loss_total"] < best_loss:
                best_loss = summary["loss_total"]

            latest_path = checkpoints_dir / "latest.pth"
            save_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                config=cfg,
                args=vars(args),
                metrics=summary,
            )

            if summary["loss_total"] == best_loss:
                save_checkpoint(
                    checkpoints_dir / "best.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_loss=best_loss,
                    config=cfg,
                    args=vars(args),
                    metrics=summary,
                )

            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    checkpoints_dir / f"epoch_{epoch + 1:03d}.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_loss=best_loss,
                    config=cfg,
                    args=vars(args),
                    metrics=summary,
                )

            log(
                f"finished epoch {epoch + 1}/{args.epochs} in {format_seconds(epoch_time)} "
                f"loss={summary['loss_total']:.4f} "
                f"(loc={summary['loss_loc']:.4f}, conf={summary['loss_conf']:.4f}, "
                f"landm={summary['loss_landm']:.4f}) best={best_loss:.4f}"
            )

    except KeyboardInterrupt:
        interrupt_path = checkpoints_dir / "interrupt.pth"
        save_checkpoint(
            interrupt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=max(current_epoch, 0),
            global_step=global_step,
            best_loss=best_loss,
            config=cfg,
            args=vars(args),
            metrics={"status": "interrupted"},
        )
        log(f"training interrupted, checkpoint saved to {interrupt_path}")
        raise


if __name__ == "__main__":
    main()
