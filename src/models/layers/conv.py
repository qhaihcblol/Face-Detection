import torch
from torch import nn
from typing import Any, Callable


class Conv2dNormActivation(nn.Sequential):
    """Convolutional block, consists of nn.Conv2d, nn.BatchNorm2d, nn.ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        norm_layer: Callable[..., torch.nn.Module] | None = torch.nn.BatchNorm2d,
        activation_layer: Callable[..., torch.nn.Module] | None = torch.nn.LeakyReLU,
        dilation: int = 1,
        inplace: bool | None = True,
        negative_slope: float | None = None,
        bias: bool = False,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params: dict[str, Any] = {} if inplace is None else {"inplace": inplace}
            if negative_slope is not None:
                params["negative_slope"] = negative_slope
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels
