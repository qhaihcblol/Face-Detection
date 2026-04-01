from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _reset_match_targets(
    loc_t: Tensor,
    conf_t: Tensor,
    landm_t: Tensor,
    idx: int,
    landm_valid_t: Tensor | None = None,
) -> None:
    """Fill one sample's match targets with background defaults."""
    loc_t[idx].zero_()
    conf_t[idx].zero_()
    landm_t[idx].zero_()
    if landm_valid_t is not None:
        landm_valid_t[idx].zero_()


def point_form(boxes: Tensor) -> Tensor:
    """Convert priors from (cx, cy, w, h) to (xmin, ymin, xmax, ymax)."""
    return torch.cat(
        (boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1
    )


def center_size(boxes: Tensor) -> Tensor:
    """Convert boxes from (xmin, ymin, xmax, ymax) to (cx, cy, w, h)."""
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]), 1)


def intersect(box_a: Tensor, box_b: Tensor) -> Tensor:
    """Compute pairwise intersection areas between two box collections."""
    max_xy = torch.minimum(box_a[:, None, 2:], box_b[None, :, 2:])
    min_xy = torch.maximum(box_a[:, None, :2], box_b[None, :, :2])
    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a: Tensor, box_b: Tensor) -> Tensor:
    """Compute pairwise IoU between two box collections."""
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))[:, None]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return inter / torch.clamp(union, min=1e-12)


def matrix_iou(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """NumPy IoU helper used by data augmentation pipelines."""
    top_left = np.maximum(box_a[:, None, :2], box_b[None, :, :2])
    bottom_right = np.minimum(box_a[:, None, 2:], box_b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, a_min=0, a_max=None)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))[:, None]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))[None, :]
    union = area_a + area_b - inter_area
    return inter_area / np.clip(union, a_min=1e-12, a_max=None)


def matrix_iof(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """NumPy intersection-over-foreground helper used by random cropping."""
    top_left = np.maximum(box_a[:, None, :2], box_b[None, :, :2])
    bottom_right = np.minimum(box_a[:, None, 2:], box_b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, a_min=0, a_max=None)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))[:, None]
    return inter_area / np.clip(area_a, a_min=1e-12, a_max=None)


def match(
    threshold: float,
    truths: Tensor,
    priors: Tensor,
    variances: list[float] | tuple[float, float],
    labels: Tensor,
    landms: Tensor,
    loc_t: Tensor,
    conf_t: Tensor,
    landm_t: Tensor,
    idx: int,
    landm_valid_t: Tensor | None = None,
    best_prior_iou_floor: float = 0.2,
) -> None:
    """
    Match each prior with the best ground-truth box, then encode regression targets.

    ``truths`` is expected in point-form coordinates and ``priors`` in center-size
    format. Both must be normalized to the same image scale.

    ``best_prior_iou_floor`` keeps us from force-matching a ground truth whose best
    prior is still a very poor fit. Set it to ``0.0`` if you want SSD-style forced
    matching for every GT.

    ``landm_valid_t`` can be provided to preserve which matched priors have valid
    landmark supervision. It may be shaped per-anchor, per-point, or per-coordinate.
    This is needed because datasets like WIDER FACE encode missing landmarks with
    ``-1`` coordinates.
    """
    num_priors = priors.size(0)

    if truths.numel() == 0:
        _reset_match_targets(loc_t, conf_t, landm_t, idx, landm_valid_t)
        return

    overlaps = jaccard(truths, point_form(priors))

    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    best_prior_idx = best_prior_idx.squeeze(1)
    best_prior_overlap = best_prior_overlap.squeeze(1)
    best_truth_idx = best_truth_idx.squeeze(0)
    best_truth_overlap = best_truth_overlap.squeeze(0)

    valid_gt = best_prior_overlap >= best_prior_iou_floor
    valid_gt_idx = torch.nonzero(valid_gt, as_tuple=False).squeeze(1)
    best_prior_idx = best_prior_idx[valid_gt]
    if best_prior_idx.numel() == 0:
        _reset_match_targets(loc_t, conf_t, landm_t, idx, landm_valid_t)
        return

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for gt_idx, prior_idx in zip(valid_gt_idx, best_prior_idx):
        best_truth_idx[prior_idx] = gt_idx

    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx].clone()
    conf[best_truth_overlap < threshold] = 0

    matched_landms = landms[best_truth_idx]
    matched_landms_points = matched_landms.view(-1, 5, 2)
    point_valid = matched_landms_points.ge(0).all(dim=2)
    coord_valid = point_valid.unsqueeze(-1).expand(-1, -1, 2).reshape(-1, 10)
    anchor_valid = point_valid.all(dim=1)
    loc = encode(matches, priors, variances)
    landm = encode_landm(matched_landms, priors, variances)
    loc[conf <= 0] = 0
    landm[~((conf > 0).unsqueeze(1) & coord_valid)] = 0

    loc_t[idx] = loc
    conf_t[idx] = conf.to(conf_t.dtype)
    landm_t[idx] = landm
    if landm_valid_t is not None:
        if landm_valid_t.dim() == 2:
            landm_valid_t[idx] = (anchor_valid & (conf > 0)).to(landm_valid_t.dtype)
        elif landm_valid_t.dim() == 3 and landm_valid_t.size(-1) == 5:
            landm_valid_t[idx] = (point_valid & (conf > 0).unsqueeze(1)).to(landm_valid_t.dtype)
        elif landm_valid_t.dim() == 3 and landm_valid_t.size(-1) == 10:
            landm_valid_t[idx] = (coord_valid & (conf > 0).unsqueeze(1)).to(landm_valid_t.dtype)
        else:
            raise ValueError(
                "landm_valid_t must have shape (batch, priors), (batch, priors, 5), "
                "or (batch, priors, 10)"
            )


def encode(
    matched: Tensor, priors: Tensor, variances: list[float] | tuple[float, float]
) -> Tensor:
    """Encode matched boxes relative to prior centers and sizes."""
    centers = (matched[:, :2] + matched[:, 2:]) / 2
    g_cxcy = (centers - priors[:, :2]) / (variances[0] * priors[:, 2:])

    sizes = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(torch.clamp(sizes, min=1e-12)) / variances[1]
    return torch.cat((g_cxcy, g_wh), 1)


def encode_landm(
    matched: Tensor, priors: Tensor, variances: list[float] | tuple[float, float]
) -> Tensor:
    """Encode five facial landmarks relative to each prior."""
    matched = matched.view(-1, 5, 2)
    priors_xy = priors[:, :2].unsqueeze(1).expand(-1, 5, -1)
    priors_wh = priors[:, 2:].unsqueeze(1).expand(-1, 5, -1)

    encoded = (matched - priors_xy) / (variances[0] * priors_wh)
    return encoded.reshape(-1, 10)


def decode(
    loc: Tensor, priors: Tensor, variances: list[float] | tuple[float, float]
) -> Tensor:
    """Decode predicted box deltas back to point-form boxes."""
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(
    pre: Tensor, priors: Tensor, variances: list[float] | tuple[float, float]
) -> Tensor:
    """Decode predicted landmark deltas back to normalized coordinates."""
    decoded = pre.view(-1, 5, 2) * variances[0] * priors[:, 2:].unsqueeze(1)
    decoded += priors[:, :2].unsqueeze(1)
    return decoded.reshape(-1, 10)


def log_sum_exp(x: Tensor) -> Tensor:
    """Numerically stable log-sum-exp used by hard negative mining."""
    x_max = x.detach().max(dim=1, keepdim=True).values
    return torch.log(torch.sum(torch.exp(x - x_max), dim=1, keepdim=True)) + x_max
