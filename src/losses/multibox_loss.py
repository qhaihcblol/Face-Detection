from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..detection.box_utils import log_sum_exp, match

__all__ = ["MultiBoxLoss"]


class MultiBoxLoss(nn.Module):
    """
    RetinaFace/SSD-style multibox loss with landmark visibility masking.

    Expected predictions:
    - ``loc_data``: ``(batch, num_priors, 4)``
    - ``conf_data``: ``(batch, num_priors, num_classes)``
    - ``landm_data``: ``(batch, num_priors, 10)``

    Supported target formats per image:
    - ``Tensor[N, 15]``: ``[x1, y1, x2, y2, lm10, label]``
    - ``Tensor[N, 20]``: ``[x1, y1, x2, y2, lm10, label, valid5]``
    - ``Tensor[N, 25]``: ``[x1, y1, x2, y2, lm10, label, valid10]``
    - ``dict`` with keys:
      ``boxes``, ``labels`` (optional), ``landms``/``landmarks`` (optional),
      ``landm_valid``/``landmark_valid`` (optional)

    Boxes must already be in point-form and normalized to the same scale as priors.
    """

    def __init__(
        self,
        cfg: Mapping[str, object],
        num_classes: int = 2,
        overlap_thresh: float = 0.30,
        neg_pos_ratio: int = 7,
        best_prior_iou_floor: float = 0.1,
        landm_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if neg_pos_ratio <= 0:
            raise ValueError("neg_pos_ratio must be positive")

        variance = cfg.get("variance")
        if not isinstance(variance, Sequence) or len(variance) != 2:
            raise ValueError("cfg['variance'] must be a sequence of length 2")

        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.best_prior_iou_floor = best_prior_iou_floor
        self.variance = (float(variance[0]), float(variance[1]))
        self.loc_weight = float(cfg.get("loc_weight", 1.0))
        self.landm_weight = float(landm_weight)

    def forward(
        self,
        predictions: tuple[Tensor, Tensor, Tensor],
        priors: Tensor,
        targets: Sequence[Tensor | Mapping[str, Tensor]],
    ) -> tuple[Tensor, Tensor, Tensor]:
        loc_data, conf_data, landm_data = predictions

        if loc_data.dim() != 3 or loc_data.size(-1) != 4:
            raise ValueError("loc_data must have shape (batch, num_priors, 4)")
        if conf_data.dim() != 3 or conf_data.size(-1) != self.num_classes:
            raise ValueError(
                "conf_data must have shape (batch, num_priors, num_classes)"
            )
        if landm_data.dim() != 3 or landm_data.size(-1) != 10:
            raise ValueError("landm_data must have shape (batch, num_priors, 10)")

        if priors.dim() == 3:
            if priors.size(0) != 1:
                raise ValueError(
                    "priors must have shape (num_priors, 4) or (1, num_priors, 4)"
                )
            priors = priors.squeeze(0)
        if priors.dim() != 2 or priors.size(-1) != 4:
            raise ValueError("priors must have shape (num_priors, 4)")

        batch_size, num_priors, _ = loc_data.shape
        if len(targets) != batch_size:
            raise ValueError("targets length must match batch size")
        if conf_data.size(1) != num_priors or landm_data.size(1) != num_priors:
            raise ValueError("prediction heads must agree on num_priors")
        if priors.size(0) != num_priors:
            raise ValueError("priors count must match prediction num_priors")

        device = loc_data.device
        priors = priors.to(device=device, dtype=loc_data.dtype).detach()

        with torch.no_grad():
            loc_t = loc_data.new_zeros((batch_size, num_priors, 4))
            conf_t = torch.zeros(
                (batch_size, num_priors), dtype=torch.long, device=device
            )
            landm_t = landm_data.new_zeros((batch_size, num_priors, 10))
            landm_valid_t = torch.zeros(
                (batch_size, num_priors, 10), dtype=torch.bool, device=device
            )

            for batch_idx, target in enumerate(targets):
                truths, labels, landms, landm_valid = self._unpack_target(
                    target=target,
                    device=device,
                    box_dtype=loc_data.dtype,
                )
                landms_for_match = landms.clone()
                landms_for_match[~landm_valid] = -1

                match(
                    threshold=self.threshold,
                    truths=truths,
                    priors=priors,
                    variances=self.variance,
                    labels=labels,
                    landms=landms_for_match,
                    loc_t=loc_t,
                    conf_t=conf_t,
                    landm_t=landm_t,
                    idx=batch_idx,
                    landm_valid_t=landm_valid_t,
                    best_prior_iou_floor=self.best_prior_iou_floor,
                )

        positive_mask = conf_t > 0
        num_pos_per_image = positive_mask.sum(dim=1, keepdim=True)
        num_pos_total = num_pos_per_image.sum().clamp(min=1).to(loc_data.dtype)

        loss_loc = self._localization_loss(
            loc_data=loc_data,
            loc_t=loc_t,
            positive_mask=positive_mask,
            normalizer=num_pos_total,
        )
        loss_conf = self._classification_loss(
            conf_data=conf_data,
            conf_t=conf_t,
            positive_mask=positive_mask,
            num_pos_per_image=num_pos_per_image,
            normalizer=num_pos_total,
        )
        loss_landm = self._landmark_loss(
            landm_data=landm_data,
            landm_t=landm_t,
            landm_valid_t=landm_valid_t,
        )

        return loss_loc, loss_conf, loss_landm

    def combine_losses(
        self, loss_loc: Tensor, loss_conf: Tensor, loss_landm: Tensor
    ) -> Tensor:
        """Combine normalized losses using the configured weights."""
        return self.loc_weight * loss_loc + loss_conf + self.landm_weight * loss_landm

    def _localization_loss(
        self,
        loc_data: Tensor,
        loc_t: Tensor,
        positive_mask: Tensor,
        normalizer: Tensor,
    ) -> Tensor:
        if not positive_mask.any():
            return loc_data.sum() * 0

        loss_loc = F.smooth_l1_loss(
            loc_data[positive_mask],
            loc_t[positive_mask],
            reduction="sum",
        )
        return loss_loc / normalizer

    def _classification_loss(
        self,
        conf_data: Tensor,
        conf_t: Tensor,
        positive_mask: Tensor,
        num_pos_per_image: Tensor,
        normalizer: Tensor,
    ) -> Tensor:
        batch_size, num_priors, _ = conf_data.shape

        with torch.no_grad():
            mining_logits = conf_data.detach().view(-1, self.num_classes)
            mining_loss = log_sum_exp(mining_logits)
            mining_loss -= mining_logits.gather(1, conf_t.view(-1, 1))
            mining_loss = mining_loss.view(batch_size, num_priors)
            mining_loss[positive_mask] = -torch.inf

            _, loss_idx = mining_loss.sort(dim=1, descending=True)
            _, idx_rank = loss_idx.sort(dim=1)

            num_neg = torch.clamp(
                self.neg_pos_ratio * num_pos_per_image,
                max=max(num_priors - 1, 0),
            )
            negative_mask = idx_rank < num_neg.expand_as(idx_rank)

        sample_mask = positive_mask | negative_mask
        if not sample_mask.any():
            return conf_data.sum() * 0

        loss_conf = F.cross_entropy(
            conf_data[sample_mask].view(-1, self.num_classes),
            conf_t[sample_mask],
            reduction="sum",
        )
        return loss_conf / normalizer

    def _landmark_loss(
        self,
        landm_data: Tensor,
        landm_t: Tensor,
        landm_valid_t: Tensor,
    ) -> Tensor:
        if not landm_valid_t.any():
            return landm_data.sum() * 0

        loss_landm = F.smooth_l1_loss(
            landm_data[landm_valid_t],
            landm_t[landm_valid_t],
            reduction="sum",
        )

        # Ten valid coordinates correspond to one fully supervised face.
        valid_face_equivalents = (
            landm_valid_t.sum().to(landm_data.dtype) / 10.0
        ).clamp(min=1.0)
        return loss_landm / valid_face_equivalents

    def _unpack_target(
        self,
        target: Tensor | Mapping[str, Tensor],
        device: torch.device,
        box_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if isinstance(target, Tensor):
            truths, labels, landms, landm_valid = self._unpack_tensor_target(target)
        elif isinstance(target, Mapping):
            truths, labels, landms, landm_valid = self._unpack_mapping_target(target)
        else:
            raise TypeError("each target must be a Tensor or mapping of tensors")

        truths = truths.to(device=device, dtype=box_dtype).reshape(-1, 4)
        labels = labels.to(device=device, dtype=torch.long).view(-1)
        landms = landms.to(device=device, dtype=box_dtype).reshape(-1, 10)
        if truths.size(0) != labels.size(0) or truths.size(0) != landms.size(0):
            raise ValueError(
                "boxes, labels, and landmarks must have the same number of rows"
            )
        landm_valid = self._expand_landmark_validity(
            landm_valid=landm_valid,
            landms=landms,
            device=device,
        )

        if truths.numel() == 0:
            return (
                truths.reshape(0, 4),
                labels.reshape(0),
                landms.reshape(0, 10),
                landm_valid.reshape(0, 10),
            )

        keep = (
            (labels > 0) & (truths[:, 2] > truths[:, 0]) & (truths[:, 3] > truths[:, 1])
        )
        truths = truths[keep]
        labels = labels[keep]
        landms = landms[keep]
        landm_valid = landm_valid[keep]

        if labels.numel() > 0 and labels.max().item() >= self.num_classes:
            raise ValueError("target labels must be in [0, num_classes - 1]")

        return truths, labels, landms, landm_valid

    def _unpack_tensor_target(
        self, target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if target.dim() != 2:
            raise ValueError("tensor targets must have shape (num_gt, num_fields)")
        if target.numel() == 0:
            empty = target.new_zeros((0,))
            return (
                target.new_zeros((0, 4)),
                empty.to(dtype=torch.long),
                target.new_zeros((0, 10)),
                target.new_zeros((0, 10), dtype=torch.bool),
            )

        num_fields = target.size(1)
        if num_fields not in {15, 20, 25}:
            raise ValueError(
                "tensor targets must have 15, 20, or 25 columns: "
                "[box4, landm10, label, optional validity]"
            )

        truths = target[:, :4]
        landms = target[:, 4:14]
        labels = target[:, 14]
        landm_valid: Tensor | None = None
        if num_fields == 20:
            landm_valid = target[:, 15:20]
        elif num_fields == 25:
            landm_valid = target[:, 15:25]

        return truths, labels, landms, landm_valid

    def _unpack_mapping_target(
        self, target: Mapping[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        truths = target.get("boxes")
        if truths is None:
            truths = target.get("truths")
        if truths is None:
            raise KeyError("target mapping must contain 'boxes' or 'truths'")

        landms = target.get("landms")
        if landms is None:
            landms = target.get("landmarks")

        labels = target.get("labels")
        if labels is None:
            labels = torch.ones(
                (truths.size(0),), device=truths.device, dtype=truths.dtype
            )

        landm_valid = target.get("landm_valid")
        if landm_valid is None:
            landm_valid = target.get("landmark_valid")
        if landm_valid is None:
            landm_valid = target.get("landmarks_valid")

        if landms is None:
            landms = torch.full(
                (truths.size(0), 10), -1, device=truths.device, dtype=truths.dtype
            )

        return truths, labels, landms, landm_valid

    def _expand_landmark_validity(
        self,
        landm_valid: Tensor | None,
        landms: Tensor,
        device: torch.device,
    ) -> Tensor:
        if landm_valid is None:
            point_valid = landms.view(-1, 5, 2).ge(0).all(dim=2)
            return point_valid.unsqueeze(-1).expand(-1, -1, 2).reshape(-1, 10)

        landm_valid = landm_valid.to(device=device)
        if landm_valid.dim() != 2:
            raise ValueError(
                "landmark validity must have shape (num_gt, 5) or (num_gt, 10)"
            )
        if landm_valid.size(0) != landms.size(0):
            raise ValueError("landmark validity rows must match number of targets")
        if landm_valid.size(1) == 5:
            point_valid = landm_valid > 0
            return point_valid.unsqueeze(-1).expand(-1, -1, 2).reshape(-1, 10)
        if landm_valid.size(1) == 10:
            return landm_valid > 0
        raise ValueError("landmark validity must have 5 or 10 columns")
