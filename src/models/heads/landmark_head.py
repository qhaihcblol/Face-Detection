import torch
from torch import Tensor, nn


class LandmarkHead(nn.Module):
    def __init__(
        self, in_channels: int = 512, num_anchors: int = 2, fpn_num: int = 3
    ) -> None:
        super().__init__()
        self.landmark_head = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    num_anchors * 10,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                )
                for _ in range(fpn_num)
            ]
        )

    def forward(self, x: list[Tensor]) -> Tensor:
        output_tensors: list[Tensor] = []
        for feature, layer in zip(x, self.landmark_head):
            output_tensors.append(layer(feature).permute(0, 2, 3, 1).contiguous())

        landmarks = torch.cat(
            [out.view(out.shape[0], -1, 10) for out in output_tensors], dim=1
        )
        return landmarks
