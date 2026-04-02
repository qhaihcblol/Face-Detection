from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import Tensor

from ..detection.box_utils import matrix_iof

__all__ = [
    "RetinaFaceTrainTransform",
    "RetinaFaceEvalTransform",
]


def _normalize_bgr_mean(
    bgr_mean: tuple[float, float, float] | tuple[float, ...],
) -> tuple[float, float, float]:
    if len(bgr_mean) != 3:
        raise ValueError("bgr_mean must have exactly 3 values (B, G, R)")
    return float(bgr_mean[0]), float(bgr_mean[1]), float(bgr_mean[2])


class RetinaFaceTrainTransform:
    """
    Standard RetinaFace train-time preprocessing and augmentation.

    Pipeline:
    1. photometric distortion
    2. random square crop
    3. random horizontal flip
    4. pad to square
    5. resize to ``image_size``
    6. convert RGB to BGR and subtract mean
    """

    def __init__(
        self,
        image_size: int = 640,
        bgr_mean: tuple[float, float, float] = (104.0, 117.0, 123.0),
        min_face_size: float = 0.0,
        crop_scales: tuple[float, ...] = (0.3, 0.45, 0.6, 0.8, 1.0),
        max_crop_trials: int = 250,
        horizontal_flip_prob: float = 0.5,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if max_crop_trials <= 0:
            raise ValueError("max_crop_trials must be positive")

        self.image_size = int(image_size)
        self.bgr_mean = _normalize_bgr_mean(bgr_mean)
        self.min_face_size = float(min_face_size)
        self.crop_scales = tuple(float(scale) for scale in crop_scales)
        self.max_crop_trials = int(max_crop_trials)
        self.horizontal_flip_prob = float(horizontal_flip_prob)

    def __call__(
        self, image: Tensor | Image.Image | np.ndarray, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Any]]:
        image_np = _to_numpy_rgb_image(image)
        target_np = _target_to_numpy(target)

        image_np = _photometric_distort(image_np)
        image_np, target_np = _random_square_crop(
            image=image_np,
            target=target_np,
            image_size=self.image_size,
            min_face_size=self.min_face_size,
            crop_scales=self.crop_scales,
            max_trials=self.max_crop_trials,
        )
        image_np, target_np = _random_horizontal_flip(
            image=image_np,
            target=target_np,
            probability=self.horizontal_flip_prob,
        )
        image_np = _pad_to_square(image_np, fill=_bgr_mean_to_rgb_fill(self.bgr_mean))
        image_np, target_np = _resize_with_targets(
            image=image_np,
            target=target_np,
            output_size=self.image_size,
        )

        image_tensor = _to_bgr_mean_subtracted_tensor(image_np, self.bgr_mean)
        return image_tensor, _target_from_numpy(target, target_np)


class RetinaFaceEvalTransform:
    """RetinaFace evaluation/inference preprocessing without random augmentation."""

    def __init__(
        self,
        image_size: int = 640,
        bgr_mean: tuple[float, float, float] = (104.0, 117.0, 123.0),
        pad_to_square: bool = True,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.image_size = int(image_size)
        self.bgr_mean = _normalize_bgr_mean(bgr_mean)
        self.pad_to_square = bool(pad_to_square)

    def __call__(
        self, image: Tensor | Image.Image | np.ndarray, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Any]]:
        image_np = _to_numpy_rgb_image(image)
        target_np = _target_to_numpy(target)

        if self.pad_to_square:
            image_np = _pad_to_square(
                image_np, fill=_bgr_mean_to_rgb_fill(self.bgr_mean)
            )

        image_np, target_np = _resize_with_targets(
            image=image_np,
            target=target_np,
            output_size=self.image_size,
        )

        image_tensor = _to_bgr_mean_subtracted_tensor(image_np, self.bgr_mean)
        return image_tensor, _target_from_numpy(target, target_np)


def _target_to_numpy(target: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            _to_numpy_copy(value)
            if key
            in {"boxes", "labels", "landmarks", "landmark_valid", "annotation_score"}
            else value
        )
        for key, value in target.items()
    }


def _target_from_numpy(
    original: dict[str, Any], target_np: dict[str, Any]
) -> dict[str, Any]:
    converted = dict(original)
    for key, value in target_np.items():
        if key == "boxes":
            converted[key] = torch.from_numpy(value.astype(np.float32, copy=False))
        elif key == "labels":
            converted[key] = torch.from_numpy(value.astype(np.int64, copy=False))
        elif key == "landmarks":
            converted[key] = torch.from_numpy(value.astype(np.float32, copy=False))
        elif key == "landmark_valid":
            converted[key] = torch.from_numpy(value.astype(np.bool_, copy=False))
        elif key == "annotation_score":
            converted[key] = torch.from_numpy(value.astype(np.float32, copy=False))
        else:
            converted[key] = value
    return converted


def _to_numpy_copy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    raise TypeError(f"unsupported target value type: {type(value)!r}")


def _to_numpy_rgb_image(image: Tensor | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return array.copy()

    if isinstance(image, Tensor):
        image = image.detach().cpu()
        if image.dim() != 3:
            raise ValueError("tensor image must have shape (C, H, W) or (H, W, C)")
        if image.shape[0] in {1, 3}:
            array = image.permute(1, 2, 0).numpy()
        else:
            array = image.numpy()
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array.copy()

    if isinstance(image, np.ndarray):
        if image.ndim != 3:
            raise ValueError("image array must have shape (H, W, C)")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image.copy()

    raise TypeError(f"unsupported image type: {type(image)!r}")


def _photometric_distort(image: np.ndarray) -> np.ndarray:
    pil_image = Image.fromarray(image)

    if random.random() < 0.5:
        pil_image = ImageEnhance.Brightness(pil_image).enhance(
            random.uniform(0.75, 1.25)
        )

    contrast_first = random.random() < 0.5
    if contrast_first and random.random() < 0.5:
        pil_image = ImageEnhance.Contrast(pil_image).enhance(random.uniform(0.75, 1.25))

    if random.random() < 0.5:
        pil_image = ImageEnhance.Color(pil_image).enhance(random.uniform(0.75, 1.25))

    if random.random() < 0.5:
        hsv = np.asarray(pil_image.convert("HSV"), dtype=np.uint8).copy()
        hue_delta = random.randint(-18, 18)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue_delta) % 255
        pil_image = Image.frombuffer(
            "HSV",
            pil_image.size,
            hsv.tobytes(),
            "raw",
            "HSV",
            0,
            1,
        ).convert("RGB")

    if not contrast_first and random.random() < 0.5:
        pil_image = ImageEnhance.Contrast(pil_image).enhance(random.uniform(0.75, 1.25))

    return np.asarray(pil_image, dtype=np.uint8)


def _random_square_crop(
    image: np.ndarray,
    target: dict[str, Any],
    image_size: int,
    min_face_size: float,
    crop_scales: tuple[float, ...],
    max_trials: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    boxes = target["boxes"]
    if boxes.size == 0:
        return image, target

    height, width, _ = image.shape
    short_side = min(height, width)
    roi = np.zeros((1, 4), dtype=np.float32)

    for _ in range(max_trials):
        scale = random.choice(crop_scales)
        crop_size = int(scale * short_side)
        crop_size = max(1, min(crop_size, height, width))

        left = 0 if width == crop_size else random.randint(0, width - crop_size)
        top = 0 if height == crop_size else random.randint(0, height - crop_size)
        roi[0] = np.asarray(
            [left, top, left + crop_size, top + crop_size], dtype=np.float32
        )

        fully_inside = (
            matrix_iof(boxes.astype(np.float32, copy=False), roi)[:, 0] >= 1.0
        )
        if not fully_inside.any():
            continue

        boxes_t = boxes[fully_inside].copy()
        labels_t = target["labels"][fully_inside].copy()
        landmarks_t = target["landmarks"][fully_inside].copy()
        landmark_valid_t = target["landmark_valid"][fully_inside].copy()
        annotation_score_t = target["annotation_score"][fully_inside].copy()

        centers = (boxes_t[:, :2] + boxes_t[:, 2:]) / 2
        center_inside = (
            (centers[:, 0] > left)
            & (centers[:, 0] < left + crop_size)
            & (centers[:, 1] > top)
            & (centers[:, 1] < top + crop_size)
        )
        if not center_inside.any():
            continue

        boxes_t = boxes_t[center_inside]
        labels_t = labels_t[center_inside]
        landmarks_t = landmarks_t[center_inside]
        landmark_valid_t = landmark_valid_t[center_inside]
        annotation_score_t = annotation_score_t[center_inside]

        boxes_t[:, 0::2] -= left
        boxes_t[:, 1::2] -= top
        boxes_t[:, 0::2] = np.clip(boxes_t[:, 0::2], 0, crop_size)
        boxes_t[:, 1::2] = np.clip(boxes_t[:, 1::2], 0, crop_size)

        landmarks_t = _translate_landmarks(
            landmarks_t,
            landmark_valid_t,
            dx=-left,
            dy=-top,
        )

        resized_face_sizes = (
            np.minimum(
                boxes_t[:, 2] - boxes_t[:, 0],
                boxes_t[:, 3] - boxes_t[:, 1],
            )
            * float(image_size)
            / float(crop_size)
        )
        keep = resized_face_sizes >= min_face_size
        if not keep.any():
            continue

        cropped_image = image[top : top + crop_size, left : left + crop_size]
        cropped_target = dict(target)
        cropped_target["boxes"] = boxes_t[keep]
        cropped_target["labels"] = labels_t[keep]
        cropped_target["landmarks"] = landmarks_t[keep]
        cropped_target["landmark_valid"] = landmark_valid_t[keep]
        cropped_target["annotation_score"] = annotation_score_t[keep]
        return cropped_image, cropped_target

    return image, target


def _random_horizontal_flip(
    image: np.ndarray,
    target: dict[str, Any],
    probability: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if random.random() >= probability:
        return image, target

    flipped_image = np.ascontiguousarray(image[:, ::-1])
    height, width, _ = flipped_image.shape

    boxes = target["boxes"].copy()
    if boxes.size > 0:
        old_x1 = boxes[:, 0].copy()
        old_x2 = boxes[:, 2].copy()
        boxes[:, 0] = width - old_x2
        boxes[:, 2] = width - old_x1

    landmarks = target["landmarks"].copy().reshape(-1, 5, 2)
    landmark_valid = target["landmark_valid"].copy()
    if landmarks.size > 0:
        valid_mask = landmark_valid
        landmarks[:, :, 0] = np.where(valid_mask, width - landmarks[:, :, 0], -1.0)
        reorder = np.asarray([1, 0, 2, 4, 3], dtype=np.int64)
        landmarks = landmarks[:, reorder, :]
        landmark_valid = landmark_valid[:, reorder]
        landmarks = np.where(landmark_valid[:, :, None], landmarks, -1.0)

    flipped_target = dict(target)
    flipped_target["boxes"] = boxes
    flipped_target["landmarks"] = landmarks.reshape(-1, 10)
    flipped_target["landmark_valid"] = landmark_valid
    return flipped_image, flipped_target


def _pad_to_square(
    image: np.ndarray,
    fill: tuple[float, float, float],
) -> np.ndarray:
    height, width, channels = image.shape
    if height == width:
        return image

    side = max(height, width)
    canvas = np.empty((side, side, channels), dtype=np.uint8)
    canvas[...] = np.asarray(fill, dtype=np.uint8)
    canvas[:height, :width] = image
    return canvas


def _bgr_mean_to_rgb_fill(
    bgr_mean: tuple[float, float, float],
) -> tuple[float, float, float]:
    return float(bgr_mean[2]), float(bgr_mean[1]), float(bgr_mean[0])


def _resize_with_targets(
    image: np.ndarray,
    target: dict[str, Any],
    output_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    original_height, original_width = image.shape[:2]
    resized_image = np.asarray(
        Image.fromarray(image).resize(
            (output_size, output_size), resample=Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )

    scale_x = float(output_size) / float(original_width)
    scale_y = float(output_size) / float(original_height)

    resized_target = dict(target)
    boxes = target["boxes"].copy()
    if boxes.size > 0:
        boxes[:, 0::2] *= scale_x
        boxes[:, 1::2] *= scale_y
    resized_target["boxes"] = boxes

    landmarks = target["landmarks"].copy()
    landmark_valid = target["landmark_valid"].copy()
    if landmarks.size > 0:
        landmarks = _scale_landmarks(
            landmarks,
            landmark_valid,
            scale_x=scale_x,
            scale_y=scale_y,
        )
    resized_target["landmarks"] = landmarks
    resized_target["landmark_valid"] = landmark_valid
    return resized_image, resized_target


def _translate_landmarks(
    landmarks: np.ndarray,
    landmark_valid: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    translated = landmarks.copy().reshape(-1, 5, 2)
    if translated.size == 0:
        return landmarks.copy()

    valid_mask = landmark_valid
    translated[:, :, 0] = np.where(valid_mask, translated[:, :, 0] + dx, -1.0)
    translated[:, :, 1] = np.where(valid_mask, translated[:, :, 1] + dy, -1.0)
    return translated.reshape(-1, 10)


def _scale_landmarks(
    landmarks: np.ndarray,
    landmark_valid: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    scaled = landmarks.copy().reshape(-1, 5, 2)
    if scaled.size == 0:
        return landmarks.copy()

    valid_mask = landmark_valid
    scaled[:, :, 0] = np.where(valid_mask, scaled[:, :, 0] * scale_x, -1.0)
    scaled[:, :, 1] = np.where(valid_mask, scaled[:, :, 1] * scale_y, -1.0)
    return scaled.reshape(-1, 10)


def _to_bgr_mean_subtracted_tensor(
    image: np.ndarray,
    bgr_mean: tuple[float, float, float],
) -> Tensor:
    image = image.astype(np.float32, copy=False)
    image = image[:, :, ::-1]  # RGB -> BGR
    image -= np.asarray(bgr_mean, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
