from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import Tensor

__all__ = ["FaceDetectorONNX", "FaceDetectorOnnx"]


@dataclass(frozen=True)
class _PreprocessMeta:
    scale_x: float
    scale_y: float
    original_width: int
    original_height: int


class FaceDetectorONNX:
    def __init__(
        self,
        onnx_path: str | Path,
        *,
        metadata_path: str | Path | None = None,
        providers: Sequence[str] | None = None,
        image_size: int | None = None,
        bgr_mean: tuple[float, float, float] = (104.0, 117.0, 123.0),
        pad_to_square: bool = True,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.4,
        top_k: int = 5000,
        keep_top_k: int = 750,
        intra_op_num_threads: int | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        metadata = self._load_metadata(self.onnx_path, metadata_path)
        metadata_input = metadata.get("input", {}) if isinstance(metadata, dict) else {}
        metadata_thresholds = (
            metadata.get("postprocess_required", {}).get("suggested_thresholds", {})
            if isinstance(metadata, dict)
            else {}
        )

        self._ort = self._import_onnxruntime()
        self.session = self._create_session(
            onnx_path=self.onnx_path,
            providers=providers,
            intra_op_num_threads=intra_op_num_threads,
        )

        input_shape = list(self.session.get_inputs()[0].shape)
        resolved_image_size = self._resolve_image_size(
            image_size=image_size,
            metadata_input=metadata_input,
            model_input_shape=input_shape,
        )
        self.image_size = int(resolved_image_size)

        self.bgr_mean = self._resolve_bgr_mean(bgr_mean, metadata_input)
        self.pad_to_square = bool(pad_to_square)

        self.conf_threshold = float(
            metadata_thresholds.get("conf_threshold", conf_threshold)
        )
        self.nms_threshold = float(
            metadata_thresholds.get("nms_threshold", nms_threshold)
        )
        self.top_k = int(metadata_thresholds.get("top_k", top_k))
        self.keep_top_k = int(metadata_thresholds.get("keep_top_k", keep_top_k))

        self.input_name = str(self.session.get_inputs()[0].name)
        outputs = self.session.get_outputs()
        if len(outputs) != 3:
            raise ValueError(
                "ONNX model must have exactly 3 outputs: boxes, scores, landmarks"
            )

        self.output_names = [str(item.name) for item in outputs]

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
        conf_thr, nms_thr, pre_nms_top_k, post_nms_top_k = (
            self._resolve_detection_params(
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                top_k=top_k,
                keep_top_k=keep_top_k,
            )
        )

        image_rgb = self._to_numpy_rgb_image(image, assume_bgr=assume_bgr)
        input_tensor, preprocess_meta = self._preprocess(image_rgb)

        boxes, scores, landmarks = self.forward_raw(input_tensor)
        det_boxes, det_scores, det_landmarks = self._postprocess_predictions(
            boxes,
            scores,
            landmarks,
            preprocess_meta=preprocess_meta,
            conf_threshold=conf_thr,
            nms_threshold=nms_thr,
            top_k=pre_nms_top_k,
            keep_top_k=post_nms_top_k,
        )
        return self._format_detections(det_boxes, det_scores, det_landmarks)

    def forward_raw(
        self, input_tensor: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if input_tensor.dtype != np.float32:
            input_tensor = input_tensor.astype(np.float32, copy=False)

        outputs = self.session.run(
            self.output_names,
            {self.input_name: np.ascontiguousarray(input_tensor)},
        )
        if len(outputs) != 3:
            raise RuntimeError("Unexpected ONNX outputs")

        boxes, scores, landmarks = outputs

        if boxes.ndim != 3 or boxes.shape[2] != 4:
            raise ValueError(f"Invalid boxes shape: {boxes.shape}")
        if scores.ndim != 2:
            raise ValueError(f"Invalid scores shape: {scores.shape}")
        if landmarks.ndim not in {3, 4}:
            raise ValueError(f"Invalid landmarks shape: {landmarks.shape}")

        if landmarks.ndim == 3:
            if landmarks.shape[2] != 10:
                raise ValueError(f"Invalid landmarks shape: {landmarks.shape}")
            landmarks = landmarks.reshape(landmarks.shape[0], landmarks.shape[1], 5, 2)

        return boxes, scores, landmarks

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

    def _postprocess_predictions(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        landmarks: np.ndarray,
        *,
        preprocess_meta: _PreprocessMeta,
        conf_threshold: float,
        nms_threshold: float,
        top_k: int,
        keep_top_k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes_1 = np.asarray(boxes[0], dtype=np.float32)
        scores_1 = np.asarray(scores[0], dtype=np.float32)
        landmarks_1 = np.asarray(landmarks[0], dtype=np.float32)

        valid = scores_1 > conf_threshold
        if not bool(np.any(valid)):
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 5, 2), dtype=np.float32),
            )

        boxes_1 = boxes_1[valid]
        scores_1 = scores_1[valid]
        landmarks_1 = landmarks_1[valid]

        # ONNX outputs are already in pixel coordinates on model input size.
        # Only map back from model-input space to padded/original image space.
        scale_x = preprocess_meta.scale_x
        scale_y = preprocess_meta.scale_y
        boxes_1[:, 0::2] *= scale_x
        boxes_1[:, 1::2] *= scale_y
        landmarks_1[:, :, 0] *= scale_x
        landmarks_1[:, :, 1] *= scale_y

        max_x = max(float(preprocess_meta.original_width - 1), 0.0)
        max_y = max(float(preprocess_meta.original_height - 1), 0.0)
        boxes_1[:, 0::2] = np.clip(boxes_1[:, 0::2], 0.0, max_x)
        boxes_1[:, 1::2] = np.clip(boxes_1[:, 1::2], 0.0, max_y)
        landmarks_1[:, :, 0] = np.clip(landmarks_1[:, :, 0], 0.0, max_x)
        landmarks_1[:, :, 1] = np.clip(landmarks_1[:, :, 1], 0.0, max_y)

        order = np.argsort(scores_1)[::-1]
        if top_k > 0:
            order = order[:top_k]

        boxes_1 = boxes_1[order]
        scores_1 = scores_1[order]
        landmarks_1 = landmarks_1[order]

        keep = self._nms(boxes_1, scores_1, threshold=nms_threshold)
        if keep_top_k > 0:
            keep = keep[:keep_top_k]

        return boxes_1[keep], scores_1[keep], landmarks_1[keep]

    @staticmethod
    def _format_detections(
        boxes: np.ndarray,
        scores: np.ndarray,
        landmarks: np.ndarray,
    ) -> list[dict[str, Any]]:
        if boxes.size == 0:
            return []

        detections: list[dict[str, Any]] = []
        for box, score, landmark in zip(boxes, scores, landmarks):
            detections.append(
                {
                    "bbox": box.astype(np.float32, copy=False).tolist(),
                    "score": float(score),
                    "landmarks": landmark.astype(np.float32, copy=False).tolist(),
                }
            )
        return detections

    def _resolve_detection_params(
        self,
        *,
        conf_threshold: float | None,
        nms_threshold: float | None,
        top_k: int | None,
        keep_top_k: int | None,
    ) -> tuple[float, float, int, int]:
        return (
            float(self.conf_threshold if conf_threshold is None else conf_threshold),
            float(self.nms_threshold if nms_threshold is None else nms_threshold),
            int(self.top_k if top_k is None else top_k),
            int(self.keep_top_k if keep_top_k is None else keep_top_k),
        )

    def _preprocess(self, image_rgb: np.ndarray) -> tuple[np.ndarray, _PreprocessMeta]:
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
            Image.fromarray(processed).resize(
                (self.image_size, self.image_size),
                resample=bilinear,
            ),
            dtype=np.uint8,
        )

        image_bgr = resized[:, :, ::-1].astype(np.float32, copy=False)
        tensor = np.ascontiguousarray(image_bgr.transpose(2, 0, 1)[None, ...])

        preprocess_meta = _PreprocessMeta(
            scale_x=float(processed_width) / float(self.image_size),
            scale_y=float(processed_height) / float(self.image_size),
            original_width=int(original_width),
            original_height=int(original_height),
        )
        return tensor, preprocess_meta

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

            array = FaceDetectorONNX._to_uint8(array)
            if assume_bgr:
                array = array[:, :, ::-1]
            return array

        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError("numpy image must have shape (H, W, C)")
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)

            array = FaceDetectorONNX._to_uint8(image)
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

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0,), dtype=np.int64)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
        order = np.argsort(scores)[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            inter_w = np.maximum(xx2 - xx1, 0.0)
            inter_h = np.maximum(yy2 - yy1, 0.0)
            inter = inter_w * inter_h
            union = areas[i] + areas[rest] - inter
            iou = inter / np.maximum(union, 1e-12)

            order = rest[iou <= threshold]

        return np.asarray(keep, dtype=np.int64)

    @staticmethod
    def _import_onnxruntime() -> Any:
        try:
            return cast(Any, importlib.import_module("onnxruntime"))
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required. Install `onnxruntime` or `onnxruntime-gpu`."
            ) from exc

    def _create_session(
        self,
        *,
        onnx_path: Path,
        providers: Sequence[str] | None,
        intra_op_num_threads: int | None,
    ) -> Any:
        session_options = self._ort.SessionOptions()
        session_options.graph_optimization_level = (
            self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if isinstance(intra_op_num_threads, int) and intra_op_num_threads > 0:
            session_options.intra_op_num_threads = int(intra_op_num_threads)

        available = set(self._ort.get_available_providers())
        if providers is None:
            requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            requested = [str(name) for name in providers]

        resolved = [name for name in requested if name in available]
        if not resolved:
            raise RuntimeError(
                "No requested ONNX Runtime provider is available. "
                f"Requested={requested}, Available={sorted(available)}"
            )

        return self._ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=resolved,
        )

    @staticmethod
    def _load_metadata(
        onnx_path: Path,
        metadata_path: str | Path | None,
    ) -> dict[str, Any]:
        if metadata_path is None:
            candidate = onnx_path.with_suffix(".json")
            if not candidate.exists():
                return {}
            path = candidate
        else:
            path = Path(metadata_path)
            if not path.exists():
                raise FileNotFoundError(f"Metadata file not found: {path}")

        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Metadata must be a JSON object")
        return cast(dict[str, Any], raw)

    def _resolve_image_size(
        self,
        image_size: int | None,
        metadata_input: Any,
        model_input_shape: list[Any],
    ) -> int:
        if image_size is not None:
            return int(image_size)

        model_h = model_input_shape[2]
        model_w = model_input_shape[3]
        if isinstance(model_h, int) and isinstance(model_w, int) and model_h == model_w:
            return int(model_h)

        if isinstance(metadata_input, dict):
            meta_size = metadata_input.get("shape")
            if isinstance(meta_size, list) and len(meta_size) == 4:
                h = meta_size[2]
                w = meta_size[3]
                if isinstance(h, int) and isinstance(w, int) and h == w:
                    return int(h)

        raise ValueError(
            "Cannot infer image_size. Pass image_size explicitly or provide metadata json."
        )

    @staticmethod
    def _resolve_bgr_mean(
        fallback_bgr_mean: tuple[float, float, float],
        metadata_input: Any,
    ) -> tuple[float, float, float]:
        if isinstance(metadata_input, dict):
            preprocess = metadata_input.get("preprocess_in_model")
            if isinstance(preprocess, dict):
                raw = preprocess.get("subtract_bgr_mean")
                if isinstance(raw, list) and len(raw) == 3:
                    if all(isinstance(v, (int, float)) for v in raw):
                        return (float(raw[0]), float(raw[1]), float(raw[2]))

        return (
            float(fallback_bgr_mean[0]),
            float(fallback_bgr_mean[1]),
            float(fallback_bgr_mean[2]),
        )


FaceDetectorOnnx = FaceDetectorONNX
