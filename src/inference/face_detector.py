from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from ..config.config import get_config
from ..detection.box_utils import decode, decode_landm
from ..detection.prior_box import PriorBox
from ..models.retinaface import RetinaFace

__all__ = ["FaceDetector"]


class FaceDetector:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        config_name: str = "mobilenet_v2",
        device: str | torch.device | None = None,
        image_size: int | None = None,
        bgr_mean: tuple[float, float, float] = (104.0, 117.0, 123.0),
        pad_to_square: bool = True,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.4,
        top_k: int = 5000,
        keep_top_k: int = 750,
    ) -> None:
        self.device = self._resolve_device(device)
        self.cfg = get_config(config_name)
        if image_size is not None:
            self.cfg["image_size"] = int(image_size)

        self.image_size = int(self.cfg["image_size"])
        self.bgr_mean = tuple(float(value) for value in bgr_mean)
        self.pad_to_square = bool(pad_to_square)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.keep_top_k = int(keep_top_k)

        # Backbone pretraining is only relevant for training initialization.
        self.cfg["pretrain"] = False
        self.model = RetinaFace(self.cfg).to(self.device)
        self.model.eval()

        self.priors = self._build_priors().to(self.device)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        strict: bool = True,
    ) -> dict[str, Any]:
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device)

        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            raise TypeError(
                "checkpoint must be a state_dict or a checkpoint dictionary"
            )

        cleaned_state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        self.model.load_state_dict(cleaned_state_dict, strict=strict)
        self.model.eval()

        if isinstance(checkpoint, dict) and "config" in checkpoint:
            ckpt_config = checkpoint.get("config")
            if isinstance(ckpt_config, dict) and "image_size" in ckpt_config:
                ckpt_image_size = int(ckpt_config["image_size"])
                if ckpt_image_size != self.image_size:
                    self.image_size = ckpt_image_size
                    self.cfg["image_size"] = ckpt_image_size
                    self.priors = self._build_priors().to(self.device)

        return checkpoint if isinstance(checkpoint, dict) else {}

    @torch.inference_mode()
    def detect(
        self,
        image: str | Path | Image.Image | np.ndarray | Tensor,
        *,
        conf_threshold: float | None = None,
        nms_threshold: float | None = None,
        top_k: int | None = None,
        keep_top_k: int | None = None,
        assume_bgr: bool = False,
    ) -> list[dict[str, Any]]:
        conf_thr = float(
            self.conf_threshold if conf_threshold is None else conf_threshold
        )
        nms_thr = float(self.nms_threshold if nms_threshold is None else nms_threshold)
        pre_nms_top_k = int(self.top_k if top_k is None else top_k)
        post_nms_top_k = int(self.keep_top_k if keep_top_k is None else keep_top_k)

        image_rgb = self._to_numpy_rgb_image(image, assume_bgr=assume_bgr)
        input_tensor, scale = self._preprocess(image_rgb)

        loc, conf, landm = self.model(input_tensor)
        scores = conf.squeeze(0)[:, 1]

        valid = scores > conf_thr
        if valid.sum().item() == 0:
            return []

        boxes = decode(loc.squeeze(0), self.priors, self.cfg["variance"])
        landms = decode_landm(landm.squeeze(0), self.priors, self.cfg["variance"])

        boxes = boxes[valid]
        landms = landms[valid]
        scores = scores[valid]

        scale_box = torch.tensor(
            [
                float(self.image_size) * scale["x"],
                float(self.image_size) * scale["y"],
                float(self.image_size) * scale["x"],
                float(self.image_size) * scale["y"],
            ],
            dtype=boxes.dtype,
            device=boxes.device,
        )
        scale_landm = torch.tensor(
            [
                float(self.image_size) * scale["x"],
                float(self.image_size) * scale["y"],
                float(self.image_size) * scale["x"],
                float(self.image_size) * scale["y"],
                float(self.image_size) * scale["x"],
                float(self.image_size) * scale["y"],
                float(self.image_size) * scale["x"],
                float(self.image_size) * scale["y"],
                float(self.image_size) * scale["x"],
                float(self.image_size) * scale["y"],
            ],
            dtype=landms.dtype,
            device=landms.device,
        )
        boxes = boxes * scale_box
        landms = landms * scale_landm

        original_height = int(scale["original_height"])
        original_width = int(scale["original_width"])
        boxes[:, 0::2] = boxes[:, 0::2].clamp_(0.0, max(float(original_width - 1), 0.0))
        boxes[:, 1::2] = boxes[:, 1::2].clamp_(
            0.0, max(float(original_height - 1), 0.0)
        )
        landms[:, 0::2] = landms[:, 0::2].clamp_(
            0.0, max(float(original_width - 1), 0.0)
        )
        landms[:, 1::2] = landms[:, 1::2].clamp_(
            0.0, max(float(original_height - 1), 0.0)
        )

        order = torch.argsort(scores, descending=True)
        if pre_nms_top_k > 0:
            order = order[:pre_nms_top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        keep = self._nms(boxes, scores, threshold=nms_thr)
        if post_nms_top_k > 0:
            keep = keep[:post_nms_top_k]

        boxes = boxes[keep].detach().cpu().numpy()
        landms = landms[keep].detach().cpu().numpy().reshape(-1, 5, 2)
        scores = scores[keep].detach().cpu().numpy()

        detections: list[dict[str, Any]] = []
        for box, score, landmark in zip(boxes, scores, landms):
            detections.append(
                {
                    "bbox": box.tolist(),
                    "score": float(score),
                    "landmarks": landmark.tolist(),
                }
            )
        return detections

    def draw(
        self,
        image: str | Path | Image.Image | np.ndarray | Tensor,
        detections: list[dict[str, Any]],
        *,
        assume_bgr: bool = False,
        box_color: tuple[int, int, int] = (0, 255, 0),
        landmark_color: tuple[int, int, int] = (255, 0, 0),
        text_color: tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2,
        radius: int = 2,
        draw_score: bool = True,
    ) -> np.ndarray:
        image_rgb = self._to_numpy_rgb_image(image, assume_bgr=assume_bgr).copy()
        canvas = Image.fromarray(image_rgb)
        drawer = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        for det in detections:
            bbox = np.asarray(det.get("bbox", []), dtype=np.float32)
            if bbox.shape != (4,):
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox]
            for offset in range(max(int(thickness), 1)):
                drawer.rectangle(
                    [(x1 - offset, y1 - offset), (x2 + offset, y2 + offset)],
                    outline=box_color,
                )

            landmarks = np.asarray(det.get("landmarks", []), dtype=np.float32).reshape(
                -1, 2
            )
            for point in landmarks:
                px, py = float(point[0]), float(point[1])
                drawer.ellipse(
                    [(px - radius, py - radius), (px + radius, py + radius)],
                    fill=landmark_color,
                    outline=landmark_color,
                )

            if draw_score and "score" in det:
                score_text = f"{float(det['score']):.3f}"
                text_position = (x1, max(0.0, y1 - 10.0))
                drawer.text(text_position, score_text, fill=text_color, font=font)

        output = np.asarray(canvas, dtype=np.uint8)
        if assume_bgr:
            output = output[:, :, ::-1]
        return output

    def _build_priors(self) -> Tensor:
        return PriorBox(
            self.cfg,
            image_size=(self.image_size, self.image_size),
        ).forward()

    def _preprocess(self, image_rgb: np.ndarray) -> tuple[Tensor, dict[str, float]]:
        original_height, original_width = image_rgb.shape[:2]
        processed = image_rgb

        if self.pad_to_square:
            side = max(original_height, original_width)
            rgb_fill = (
                int(self.bgr_mean[2]),
                int(self.bgr_mean[1]),
                int(self.bgr_mean[0]),
            )
            canvas = np.empty((side, side, 3), dtype=np.uint8)
            canvas[...] = np.asarray(rgb_fill, dtype=np.uint8)
            canvas[:original_height, :original_width] = image_rgb
            processed = canvas

        processed_height, processed_width = processed.shape[:2]
        resampling_module = getattr(Image, "Resampling", Image)
        bilinear = getattr(resampling_module, "BILINEAR")
        resized = np.asarray(
            Image.fromarray(processed, mode="RGB").resize(
                (self.image_size, self.image_size),
                resample=bilinear,
            ),
            dtype=np.uint8,
        )

        image_bgr = resized[:, :, ::-1].astype(np.float32, copy=False)
        image_bgr -= np.asarray(self.bgr_mean, dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(image_bgr)).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).to(self.device)

        scale = {
            "x": float(processed_width) / float(self.image_size),
            "y": float(processed_height) / float(self.image_size),
            "original_width": float(original_width),
            "original_height": float(original_height),
        }
        return tensor, scale

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return resolved

    @staticmethod
    def _nms(boxes: Tensor, scores: Tensor, threshold: float) -> Tensor:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = scores.argsort(descending=True)

        keep: list[int] = []
        while order.numel() > 0:
            i = int(order[0].item())
            keep.append(i)
            if order.numel() == 1:
                break

            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])

            inter_w = (xx2 - xx1).clamp(min=0)
            inter_h = (yy2 - yy1).clamp(min=0)
            inter = inter_w * inter_h
            union = areas[i] + areas[rest] - inter
            iou = inter / torch.clamp(union, min=1e-12)

            order = rest[iou <= threshold]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    @staticmethod
    def _to_numpy_rgb_image(
        image: str | Path | Image.Image | np.ndarray | Tensor,
        *,
        assume_bgr: bool,
    ) -> np.ndarray:
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                return np.asarray(img.convert("RGB"), dtype=np.uint8)

        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

        if isinstance(image, Tensor):
            image = image.detach().cpu()
            if image.dim() != 3:
                raise ValueError("tensor image must have shape (C, H, W) or (H, W, C)")

            if image.shape[0] in {1, 3}:
                array = image.permute(1, 2, 0).numpy()
            else:
                array = image.numpy()

            if array.ndim != 3:
                raise ValueError("tensor image must be 3-dimensional")
            if array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)

            array = FaceDetector._to_uint8(array)
            if assume_bgr:
                array = array[:, :, ::-1]
            return array

        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError("numpy image must have shape (H, W, C)")
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)

            array = FaceDetector._to_uint8(image)
            if assume_bgr:
                array = array[:, :, ::-1]
            return array

        raise TypeError(f"unsupported image type: {type(image)!r}")

    @staticmethod
    def _to_uint8(array: np.ndarray) -> np.ndarray:
        if array.dtype == np.uint8:
            return array

        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size > 0 else 0.0
            if max_value <= 1.0:
                array = array * 255.0

        return np.clip(array, 0, 255).astype(np.uint8)
