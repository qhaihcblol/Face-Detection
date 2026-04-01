from __future__ import annotations

from math import ceil

import torch
from torch import Tensor


class PriorBox:
    """
    Generate RetinaFace priors in ``(cx, cy, w, h)`` format.

    The generated coordinates are normalized to the input image size so they can
    be consumed directly by the box encode/decode logic.
    """

    def __init__(self, cfg: dict, image_size: tuple[int, int]) -> None:
        self.image_size = image_size

        self.min_sizes: list[list[int]] = cfg["min_sizes"]
        self.steps: list[int] = cfg["steps"]
        self.clip: bool = cfg["clip"]

        if len(self.min_sizes) != len(self.steps):
            raise ValueError("cfg['min_sizes'] and cfg['steps'] must have the same length")

        image_height, image_width = self.image_size
        self.feature_maps: list[list[int]] = [
            [ceil(image_height / step), ceil(image_width / step)] for step in self.steps
        ]

    def forward(self) -> Tensor:
        anchors: list[float] = []
        image_height, image_width = self.image_size

        for level_idx, feature_map in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[level_idx]
            step = self.steps[level_idx]

            for row in range(feature_map[0]):
                for col in range(feature_map[1]):
                    cx = (col + 0.5) * step / image_width
                    cy = (row + 0.5) * step / image_height

                    for min_size in min_sizes:
                        scale_x = min_size / image_width
                        scale_y = min_size / image_height
                        anchors.extend([cx, cy, scale_x, scale_y])

        priors = torch.tensor(anchors, dtype=torch.float32).view(-1, 4)
        if self.clip:
            priors.clamp_(min=0.0, max=1.0)
        return priors
