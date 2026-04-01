from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["WiderFaceDataset", "widerface_collate"]


@dataclass(frozen=True)
class _WiderFaceRecord:
    relative_path: str
    boxes: np.ndarray
    landmarks: np.ndarray
    landmark_valid: np.ndarray
    annotation_score: np.ndarray


class WiderFaceDataset(Dataset[tuple[Tensor | Image.Image, dict[str, Any]]]):
    """
    WIDER FACE training dataset parser for RetinaFace-style annotations.

    The train annotation file is expected to look like:
    - ``# relative/image/path.jpg``
    - one or more rows of ``xywh + 5 * (x, y, flag) + score``

    Returned targets are compatible with ``MultiBoxLoss``:
    - ``boxes``: ``(N, 4)``
    - ``labels``: ``(N,)`` with foreground label ``1``
    - ``landmarks``: ``(N, 10)``
    - ``landmark_valid``: ``(N, 5)``

    If ``normalize_targets=True``, boxes and valid landmark coordinates are
    normalized after ``transform`` using the current image size.
    """

    def __init__(
        self,
        annotation_file: str | Path = "datasets/train/train_annotations.txt",
        image_root: str | Path | None = None,
        transform: Callable[
            [Tensor | Image.Image, dict[str, Any]],
            tuple[Tensor | Image.Image, dict[str, Any]],
        ]
        | None = None,
        normalize_targets: bool = True,
        to_tensor: bool = True,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        if not self.annotation_file.is_file():
            raise FileNotFoundError(f"annotation file not found: {self.annotation_file}")

        self.image_root = (
            Path(image_root)
            if image_root is not None
            else self.annotation_file.parent / "images"
        )
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"image root not found: {self.image_root}")

        self.transform = transform
        self.normalize_targets = normalize_targets
        self.to_tensor = to_tensor
        self.records = self._parse_train_annotations(self.annotation_file)

    @classmethod
    def from_train_split(
        cls,
        train_root: str | Path = "datasets/train",
        **kwargs: Any,
    ) -> "WiderFaceDataset":
        train_root = Path(train_root)
        return cls(
            annotation_file=train_root / "train_annotations.txt",
            image_root=train_root / "images",
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor | Image.Image, dict[str, Any]]:
        record = self.records[index]
        image_path = self.image_root / record.relative_path.lstrip("/").lstrip("./")
        if not image_path.is_file():
            raise FileNotFoundError(f"image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size

        target = {
            "boxes": torch.from_numpy(record.boxes.copy()),
            "labels": torch.ones((record.boxes.shape[0],), dtype=torch.long),
            "landmarks": torch.from_numpy(record.landmarks.copy()),
            "landmark_valid": torch.from_numpy(record.landmark_valid.copy()),
            "annotation_score": torch.from_numpy(record.annotation_score.copy()),
            "image_id": torch.tensor(index, dtype=torch.long),
            "orig_size": torch.tensor([orig_height, orig_width], dtype=torch.long),
            "image_size": torch.tensor([orig_height, orig_width], dtype=torch.long),
            "path": record.relative_path,
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        current_height, current_width = self._get_image_size(image)
        target["image_size"] = torch.tensor(
            [current_height, current_width], dtype=torch.long
        )

        if self.normalize_targets:
            self._normalize_target_in_place(target, current_height, current_width)

        if self.to_tensor:
            image = self._image_to_tensor(image)

        return image, target

    @staticmethod
    def _parse_train_annotations(annotation_file: Path) -> list[_WiderFaceRecord]:
        records: list[_WiderFaceRecord] = []
        current_path: str | None = None
        current_rows: list[np.ndarray] = []

        with annotation_file.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("#"):
                    if current_path is not None:
                        records.append(
                            WiderFaceDataset._build_record(current_path, current_rows)
                        )
                    current_path = line[1:].strip()
                    current_rows = []
                    continue

                if current_path is None:
                    raise ValueError(
                        f"invalid annotation format at line {line_no}: "
                        "expected image path starting with '#'"
                    )

                fields = line.split()
                if len(fields) < 20:
                    raise ValueError(
                        f"invalid annotation row at line {line_no}: "
                        f"expected at least 20 values, got {len(fields)}"
                    )
                current_rows.append(np.asarray(fields[:20], dtype=np.float32))

        if current_path is not None:
            records.append(WiderFaceDataset._build_record(current_path, current_rows))

        if not records:
            raise ValueError(f"no records found in {annotation_file}")

        return records

    @staticmethod
    def _build_record(relative_path: str, rows: list[np.ndarray]) -> _WiderFaceRecord:
        if not rows:
            return _WiderFaceRecord(
                relative_path=relative_path,
                boxes=np.zeros((0, 4), dtype=np.float32),
                landmarks=np.zeros((0, 10), dtype=np.float32),
                landmark_valid=np.zeros((0, 5), dtype=np.bool_),
                annotation_score=np.zeros((0,), dtype=np.float32),
            )

        boxes: list[np.ndarray] = []
        landmarks: list[np.ndarray] = []
        landmark_valid: list[np.ndarray] = []
        annotation_score: list[float] = []

        for row in rows:
            x, y, w, h = row[:4]
            if w <= 0 or h <= 0:
                continue

            boxes.append(np.asarray([x, y, x + w, y + h], dtype=np.float32))

            landmark_triplets = row[4:19].reshape(5, 3)
            landmark_xy = landmark_triplets[:, :2].reshape(10).astype(np.float32)
            valid_points = (
                (landmark_triplets[:, 2] >= 0)
                & (landmark_triplets[:, 0] >= 0)
                & (landmark_triplets[:, 1] >= 0)
            )

            landmarks.append(landmark_xy)
            landmark_valid.append(valid_points.astype(np.bool_))
            annotation_score.append(float(row[19]))

        if not boxes:
            return _WiderFaceRecord(
                relative_path=relative_path,
                boxes=np.zeros((0, 4), dtype=np.float32),
                landmarks=np.zeros((0, 10), dtype=np.float32),
                landmark_valid=np.zeros((0, 5), dtype=np.bool_),
                annotation_score=np.zeros((0,), dtype=np.float32),
            )

        return _WiderFaceRecord(
            relative_path=relative_path,
            boxes=np.stack(boxes).astype(np.float32, copy=False),
            landmarks=np.stack(landmarks).astype(np.float32, copy=False),
            landmark_valid=np.stack(landmark_valid).astype(np.bool_, copy=False),
            annotation_score=np.asarray(annotation_score, dtype=np.float32),
        )

    @staticmethod
    def _normalize_target_in_place(
        target: dict[str, Any], height: int, width: int
    ) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("image height and width must be positive")

        boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)
        landmarks = torch.as_tensor(target["landmarks"], dtype=torch.float32)
        landmark_valid = torch.as_tensor(target["landmark_valid"], dtype=torch.bool)

        if boxes.numel() > 0:
            boxes = boxes.clone()
            boxes[:, 0::2] /= float(width)
            boxes[:, 1::2] /= float(height)

        if landmarks.numel() > 0:
            landmarks = landmarks.clone()
            coord_valid = (
                landmark_valid.unsqueeze(-1).expand(-1, -1, 2).reshape(-1, 10)
            )
            scale = landmarks.new_tensor([width, height] * 5)
            landmarks[coord_valid] = landmarks[coord_valid] / scale.expand_as(landmarks)[
                coord_valid
            ]

        target["boxes"] = boxes
        target["landmarks"] = landmarks
        target["landmark_valid"] = landmark_valid

    @staticmethod
    def _get_image_size(image: Tensor | Image.Image | np.ndarray) -> tuple[int, int]:
        if isinstance(image, Image.Image):
            width, height = image.size
            return height, width

        if isinstance(image, Tensor):
            if image.dim() == 2:
                return int(image.shape[0]), int(image.shape[1])
            if image.dim() == 3:
                if image.shape[0] <= 4:
                    return int(image.shape[1]), int(image.shape[2])
                return int(image.shape[0]), int(image.shape[1])
            raise ValueError("tensor image must have shape (H, W) or (C, H, W)")

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return int(image.shape[0]), int(image.shape[1])
            if image.ndim == 3:
                if image.shape[0] <= 4:
                    return int(image.shape[1]), int(image.shape[2])
                return int(image.shape[0]), int(image.shape[1])
            raise ValueError("ndarray image must have shape (H, W) or (H, W, C)")

        raise TypeError(f"unsupported image type: {type(image)!r}")

    @staticmethod
    def _image_to_tensor(image: Tensor | Image.Image | np.ndarray) -> Tensor:
        if isinstance(image, Tensor):
            if image.is_floating_point():
                return image
            return image.float() / 255.0

        if isinstance(image, Image.Image):
            array = np.asarray(image, dtype=np.float32)
        elif isinstance(image, np.ndarray):
            array = image.astype(np.float32, copy=False)
        else:
            raise TypeError(f"unsupported image type: {type(image)!r}")

        if array.ndim == 2:
            tensor = torch.from_numpy(array).unsqueeze(0)
        elif array.ndim == 3:
            tensor = torch.from_numpy(array).permute(2, 0, 1)
        else:
            raise ValueError("image array must have shape (H, W) or (H, W, C)")

        return tensor.float() / 255.0


def widerface_collate(
    batch: list[tuple[Tensor | Image.Image, dict[str, Any]]],
) -> tuple[Tensor | list[Tensor | Image.Image], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    if all(isinstance(image, Tensor) for image in images):
        image_shapes = {tuple(image.shape) for image in images}
        if len(image_shapes) == 1:
            return torch.stack(list(images), dim=0), list(targets)
    return list(images), list(targets)
