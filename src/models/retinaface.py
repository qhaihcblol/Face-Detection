from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from .builder import build_backbone, get_layer_extractor
from .heads.bbox_head import BboxHead
from .heads.cls_head import ClassHead
from .heads.landmark_head import LandmarkHead
from .necks.fpn import FPN
from .necks.ssh import SSH


def _resolve_fpn_in_channels(backbone: nn.Module, indexes: list[int]) -> list[int]:
    features = getattr(backbone, "features", None)
    if not isinstance(features, nn.Sequential):
        raise TypeError("backbone.features must be an nn.Sequential")

    channels: list[int] = []
    for idx in indexes:
        if idx < 0 or idx >= len(features):
            raise IndexError(f"Feature index {idx} is out of range for backbone.features")

        feature_layer = features[idx]
        out_channels = getattr(feature_layer, "out_channels", None)
        if not isinstance(out_channels, int):
            raise TypeError(
                f"Feature layer at index {idx} does not expose an integer out_channels"
            )
        channels.append(out_channels)

    return channels


class RetinaFace(nn.Module):
    def __init__(self, cfg: dict | None = None) -> None:
        """
        RetinaFace model constructor.

        Args:
            cfg (dict): A configuration dictionary containing model parameters.
        """
        super().__init__()

        if cfg is None:
            raise ValueError("cfg must not be None.")

        backbone = build_backbone(cfg["name"], cfg["pretrain"])
        self.fx = get_layer_extractor(cfg, backbone)  # feature extraction

        num_anchors = 2
        out_channels = cfg["out_channel"]
        feature_indexes = cfg["return_layers"]
        fpn_in_channels = _resolve_fpn_in_channels(backbone, feature_indexes)
        self.fpn_in_channels = fpn_in_channels

        self.fpn = FPN(fpn_in_channels, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.class_head = ClassHead(
            in_channels=cfg["out_channel"], num_anchors=num_anchors, fpn_num=3
        )
        self.bbox_head = BboxHead(
            in_channels=cfg["out_channel"], num_anchors=num_anchors, fpn_num=3
        )
        self.landmark_head = LandmarkHead(
            in_channels=cfg["out_channel"], num_anchors=num_anchors, fpn_num=3
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.fx(x)
        fpn = self.fpn(out)

        # single-stage headless module
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])

        features = [feature1, feature2, feature3]

        classifications = self.class_head(features)
        bbox_regressions = self.bbox_head(features)
        landmark_regressions = self.landmark_head(features)

        if self.training:
            output = (bbox_regressions, classifications, landmark_regressions)
        else:
            output = (
                bbox_regressions,
                F.softmax(classifications, dim=-1),
                landmark_regressions,
            )
        return output
