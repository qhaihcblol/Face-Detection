#!/usr/bin/env python3
"""Convert a RetinaFace PyTorch checkpoint to an ONNX deployment model."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

# Allow running as: python scripts/convert_to_onnx.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import get_config
from src.detection.prior_box import PriorBox
from src.models.retinaface import RetinaFace


class RetinaFaceOnnxWrapper(nn.Module):
    """Deployment wrapper that decodes model outputs for easier downstream use."""

    def __init__(
        self,
        *,
        model: nn.Module,
        priors: Tensor,
        variance: tuple[float, float],
        image_size: int,
        bgr_mean: tuple[float, float, float],
    ) -> None:
        super().__init__()
        if priors.ndim != 2 or priors.shape[1] != 4:
            raise ValueError("priors must have shape (num_priors, 4)")

        self.model = model
        self.image_size = int(image_size)
        self.variance = (float(variance[0]), float(variance[1]))

        self.register_buffer("priors", priors.to(dtype=torch.float32), persistent=False)
        self.register_buffer(
            "bgr_mean",
            torch.tensor(bgr_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        box_scale = torch.tensor(
            [self.image_size, self.image_size, self.image_size, self.image_size],
            dtype=torch.float32,
        ).view(1, 1, 4)
        landm_scale = torch.tensor(
            [
                self.image_size,
                self.image_size,
                self.image_size,
                self.image_size,
                self.image_size,
                self.image_size,
                self.image_size,
                self.image_size,
                self.image_size,
                self.image_size,
            ],
            dtype=torch.float32,
        ).view(1, 1, 10)

        self.register_buffer("box_scale", box_scale, persistent=False)
        self.register_buffer("landm_scale", landm_scale, persistent=False)

    def forward(self, image_bgr: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Expected input: BGR float32 tensor in range [0, 255], shape (N, 3, H, W).
        bgr_mean = self.bgr_mean
        box_scale = self.box_scale
        landm_scale = self.landm_scale

        if not isinstance(bgr_mean, Tensor):
            raise TypeError("bgr_mean buffer is not a tensor")
        if not isinstance(box_scale, Tensor):
            raise TypeError("box_scale buffer is not a tensor")
        if not isinstance(landm_scale, Tensor):
            raise TypeError("landm_scale buffer is not a tensor")

        x = image_bgr.to(dtype=torch.float32) - bgr_mean
        loc, conf, landm = self.model(x)

        boxes = self._decode_boxes(loc) * box_scale
        scores = conf[..., 1]
        landmarks = self._decode_landmarks(landm) * landm_scale

        return boxes, scores, landmarks.reshape(image_bgr.shape[0], -1, 5, 2)

    def _decode_boxes(self, loc: Tensor) -> Tensor:
        priors = self.priors
        if not isinstance(priors, Tensor):
            raise TypeError("priors buffer is not a tensor")
        priors = priors.unsqueeze(0)
        centers = priors[..., :2] + loc[..., :2] * self.variance[0] * priors[..., 2:]
        sizes = priors[..., 2:] * torch.exp(loc[..., 2:] * self.variance[1])

        top_left = centers - sizes / 2.0
        bottom_right = centers + sizes / 2.0
        return torch.cat((top_left, bottom_right), dim=-1)

    def _decode_landmarks(self, landm: Tensor) -> Tensor:
        priors = self.priors
        if not isinstance(priors, Tensor):
            raise TypeError("priors buffer is not a tensor")
        priors = priors.unsqueeze(0)
        batch = landm.shape[0]
        num_priors = landm.shape[1]
        points = landm.reshape(batch, num_priors, 5, 2)

        decoded = points * self.variance[0] * priors[..., 2:].unsqueeze(2)
        decoded = decoded + priors[..., :2].unsqueeze(2)
        return decoded.reshape(batch, num_priors, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RetinaFace checkpoint (.pth) to ONNX with decoded boxes, "
            "scores, and landmarks."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a training checkpoint or raw model state_dict",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output ONNX path (default: <checkpoint_stem>.onnx)",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Output metadata JSON path (default: <output_stem>.json)",
    )
    parser.add_argument(
        "--config-name",
        default="mobilenet_v2",
        help="Model config name, e.g. mobilenet_v2",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Model input size. If omitted, use checkpoint config when available.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used for export, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with dynamic batch axis",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate exported model with onnxruntime if installed",
    )
    parser.add_argument(
        "--bgr-mean",
        nargs=3,
        type=float,
        metavar=("B", "G", "R"),
        default=(104.0, 117.0, 123.0),
        help="Mean used in preprocessing (default: 104 117 123)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.4,
        help="Suggested confidence threshold for runtime postprocess metadata",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.4,
        help="Suggested NMS threshold for runtime postprocess metadata",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5000,
        help="Suggested pre-NMS top-k for runtime postprocess metadata",
    )
    parser.add_argument(
        "--keep-top-k",
        type=int,
        default=750,
        help="Suggested post-NMS top-k for runtime postprocess metadata",
    )
    return parser.parse_args()


def _extract_state_dict(
    checkpoint: Any,
) -> tuple[dict[str, Tensor], dict[str, Any] | None]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a state_dict or checkpoint dictionary")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("state_dict payload is invalid")

    cleaned_state_dict: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("state_dict keys must be strings")
        if not isinstance(value, Tensor):
            raise TypeError("state_dict values must be tensors")
        new_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value

    return cleaned_state_dict, checkpoint


def _resolve_image_size(
    *,
    args_image_size: int | None,
    cfg: dict[str, Any],
    checkpoint_payload: dict[str, Any] | None,
) -> int:
    if args_image_size is not None:
        return int(args_image_size)

    if isinstance(checkpoint_payload, dict):
        ckpt_cfg = checkpoint_payload.get("config")
        if isinstance(ckpt_cfg, dict):
            ckpt_image_size = ckpt_cfg.get("image_size")
            if isinstance(ckpt_image_size, (int, float)):
                return int(ckpt_image_size)

    cfg_image_size = cfg.get("image_size")
    if not isinstance(cfg_image_size, (int, float)):
        raise TypeError("cfg['image_size'] must be numeric")
    return int(cfg_image_size)


def _resolve_variance(cfg: dict[str, Any]) -> tuple[float, float]:
    variance = cfg.get("variance")
    if not isinstance(variance, (list, tuple)) or len(variance) != 2:
        raise ValueError("cfg['variance'] must be a list/tuple with two values")
    return float(variance[0]), float(variance[1])


def _build_metadata(
    *,
    checkpoint_path: Path,
    onnx_path: Path,
    config_name: str,
    image_size: int,
    bgr_mean: tuple[float, float, float],
    num_priors: int,
    opset: int,
    dynamic_batch: bool,
    conf_threshold: float,
    nms_threshold: float,
    top_k: int,
    keep_top_k: int,
) -> dict[str, Any]:
    return {
        "model_type": "retinaface",
        "checkpoint": str(checkpoint_path),
        "onnx_path": str(onnx_path),
        "config_name": config_name,
        "opset": int(opset),
        "image_size": int(image_size),
        "num_priors": int(num_priors),
        "input": {
            "name": "input",
            "dtype": "float32",
            "layout": "NCHW",
            "color_space": "BGR",
            "shape": ["batch" if dynamic_batch else 1, 3, image_size, image_size],
            "value_range": "0..255",
            "preprocess_in_model": {
                "subtract_bgr_mean": [float(x) for x in bgr_mean],
            },
            "recommended_frame_preprocess": {
                "pad_to_square": True,
                "pad_position": "top_left",
                "pad_fill_bgr": [float(x) for x in bgr_mean],
                "resize": "bilinear",
            },
        },
        "outputs": [
            {
                "name": "boxes",
                "shape": ["batch", "num_priors", 4],
                "description": "Decoded boxes in XYXY pixel coords on model input size",
            },
            {
                "name": "scores",
                "shape": ["batch", "num_priors"],
                "description": "Face probability for class index 1",
            },
            {
                "name": "landmarks",
                "shape": ["batch", "num_priors", 5, 2],
                "description": "Decoded landmark points in pixel coords",
            },
        ],
        "postprocess_required": {
            "nms": True,
            "clip_to_image": True,
            "suggested_thresholds": {
                "conf_threshold": float(conf_threshold),
                "nms_threshold": float(nms_threshold),
                "top_k": int(top_k),
                "keep_top_k": int(keep_top_k),
            },
        },
    }


def _validate_with_onnxruntime(
    *,
    model: nn.Module,
    onnx_path: Path,
    image_size: int,
    dynamic_batch: bool,
) -> None:
    try:
        np = cast(Any, importlib.import_module("numpy"))
        ort = cast(Any, importlib.import_module("onnxruntime"))
    except ImportError:
        print("Skip validation: onnxruntime or numpy is not installed.")
        return

    model_device = next(model.parameters()).device
    batch = 2 if dynamic_batch else 1
    sample = (
        torch.rand(
            batch,
            3,
            image_size,
            image_size,
            device=model_device,
            dtype=torch.float32,
        )
        * 255.0
    )

    with torch.inference_mode():
        torch_outputs = model(sample)

    torch_np = [tensor.detach().cpu().numpy() for tensor in torch_outputs]

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    ort_outputs = session.run(None, {input_name: sample.detach().cpu().numpy()})

    atol = 1e-3
    output_names = ("boxes", "scores", "landmarks")
    for name, torch_out, ort_out in zip(output_names, torch_np, ort_outputs):
        max_abs = float(np.max(np.abs(torch_out - ort_out)))
        print(f"Validation max abs diff for {name}: {max_abs:.6f}")
        if max_abs > atol:
            raise RuntimeError(
                f"Validation failed for {name}: max abs diff {max_abs:.6f} > {atol}"
            )


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.output is None:
        onnx_path = checkpoint_path.with_suffix(".onnx")
    else:
        onnx_path = Path(args.output)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if args.metadata is None:
        metadata_path = onnx_path.with_suffix(".json")
    else:
        metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict, checkpoint_payload = _extract_state_dict(checkpoint)

    cfg = get_config(args.config_name)
    cfg["pretrain"] = False
    image_size = _resolve_image_size(
        args_image_size=args.image_size,
        cfg=cfg,
        checkpoint_payload=checkpoint_payload,
    )
    cfg["image_size"] = int(image_size)

    model = RetinaFace(cfg).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    priors = PriorBox(cfg, image_size=(image_size, image_size)).forward().to(device)
    variance = _resolve_variance(cfg)
    if len(args.bgr_mean) != 3:
        raise ValueError("--bgr-mean must have exactly 3 values: B G R")
    bgr_mean = (
        float(args.bgr_mean[0]),
        float(args.bgr_mean[1]),
        float(args.bgr_mean[2]),
    )

    export_model = RetinaFaceOnnxWrapper(
        model=model,
        priors=priors,
        variance=variance,
        image_size=image_size,
        bgr_mean=bgr_mean,
    ).to(device)
    export_model.eval()

    dummy_input = torch.zeros(
        (1, 3, image_size, image_size),
        dtype=torch.float32,
        device=device,
    )

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "boxes": {0: "batch"},
            "scores": {0: "batch"},
            "landmarks": {0: "batch"},
        }

    with torch.inference_mode():
        torch.onnx.export(
            export_model,
            (dummy_input,),
            str(onnx_path),
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["boxes", "scores", "landmarks"],
            dynamic_axes=dynamic_axes,
            opset_version=int(args.opset),
        )

    metadata = _build_metadata(
        checkpoint_path=checkpoint_path,
        onnx_path=onnx_path,
        config_name=args.config_name,
        image_size=image_size,
        bgr_mean=bgr_mean,
        num_priors=int(priors.shape[0]),
        opset=int(args.opset),
        dynamic_batch=bool(args.dynamic_batch),
        conf_threshold=float(args.conf_threshold),
        nms_threshold=float(args.nms_threshold),
        top_k=int(args.top_k),
        keep_top_k=int(args.keep_top_k),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Exported ONNX model: {onnx_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Input shape: (batch, 3, {image_size}, {image_size})")
    print(f"Num priors: {int(priors.shape[0])}")

    if args.validate:
        _validate_with_onnxruntime(
            model=export_model,
            onnx_path=onnx_path,
            image_size=image_size,
            dynamic_batch=bool(args.dynamic_batch),
        )
        print("ONNX Runtime validation passed.")


if __name__ == "__main__":
    main()
