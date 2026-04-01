#!/usr/bin/env python3
"""Count face boxes for a specific image in WIDERFace-style annotation files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterator


def iter_annotation_blocks(
    annotation_file: Path,
) -> Iterator[tuple[str, list[tuple[float, float, float, float]]]]:
    current_image: str | None = None
    current_boxes: list[tuple[float, float, float, float]] = []

    with annotation_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if current_image is not None:
                    yield current_image, current_boxes
                current_image = line[1:].strip()
                current_boxes = []
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            x, y, w, h = map(float, parts[:4])
            current_boxes.append((x, y, w, h))

    if current_image is not None:
        yield current_image, current_boxes


def count_small_faces(
    boxes: list[tuple[float, float, float, float]], threshold: float
) -> dict[str, int]:
    min_side = 0
    both_wh = 0
    sqrt_area = 0

    for _, _, w, h in boxes:
        if min(w, h) < threshold:
            min_side += 1
        if w < threshold and h < threshold:
            both_wh += 1
        if math.sqrt(w * h) < threshold:
            sqrt_area += 1

    return {
        "min_side_lt": min_side,
        "both_w_h_lt": both_wh,
        "sqrt_area_lt": sqrt_area,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count faces for one image in annotation file"
    )
    parser.add_argument(
        "--annotations",
        default="datasets/train/train_annotations.txt",
        help="Path to annotation txt file",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Image key in annotation file, e.g. 0--Parade/0_Parade_marchingband_1_6.jpg",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=16.0,
        help="Threshold used to define small faces",
    )

    args = parser.parse_args()

    annotation_file = Path(args.annotations)
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    target = args.image.strip()

    for image_key, boxes in iter_annotation_blocks(annotation_file):
        if image_key != target:
            continue

        counts = count_small_faces(boxes, args.threshold)
        total = len(boxes)

        print(f"image={image_key}")
        print(f"total_faces={total}")
        print(f"threshold={args.threshold}")
        print(f"small_faces_min_side_lt_threshold={counts['min_side_lt']}")
        print(f"small_faces_both_w_h_lt_threshold={counts['both_w_h_lt']}")
        print(f"small_faces_sqrt_area_lt_threshold={counts['sqrt_area_lt']}")
        return

    raise ValueError(f"Image key not found in annotation file: {target}")


if __name__ == "__main__":
    main()
