from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from ..layers.conv import Conv2dNormActivation

__all__ = ["SSH"]

class SSH(nn.Module):
    """
    SSH (Single Stage Headless) Module for feature extraction.
    Combines 3x3, 5x5, and 7x7 convolutions with batch normalization and optional LeakyReLU activations.
    """

    def __init__(self, in_channel: int, out_channels: int) -> None:
        """
        Initializes the SSH module.

        Args:
            in_channel (int): Number of input channels.
            out_channels (int): Number of output channels, must be divisible by 4.
        """
        super().__init__()

        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4.")
        leaky = 0.1 if out_channels <= 64 else 0

        # 3x3 Convolution branch
        self.conv3X3 = Conv2dNormActivation(
            in_channel, out_channels // 2, kernel_size=3, activation_layer=None
        )

        # 5x5 Convolution branch
        self.conv5X5_1 = Conv2dNormActivation(
            in_channel, out_channels // 4, kernel_size=3, negative_slope=leaky
        )
        self.conv5X5_2 = Conv2dNormActivation(
            out_channels // 4, out_channels // 4, kernel_size=3, activation_layer=None
        )

        # 7x7 Convolution branch
        self.conv7X7_2 = Conv2dNormActivation(
            out_channels // 4, out_channels // 4, kernel_size=3, negative_slope=leaky
        )
        self.conv7x7_3 = Conv2dNormActivation(
            out_channels // 4, out_channels // 4, kernel_size=3, activation_layer=None
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SSH module.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output feature map after applying SSH operations and ReLU activation.
        """
        conv3X3 = self.conv3X3(x)
        conv5X5_1 = self.conv5X5_1(x)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(self.conv7X7_2(conv5X5_1))

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        return F.relu(out, inplace=True)
