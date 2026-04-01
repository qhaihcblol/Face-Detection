from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ..layers.conv import Conv2dNormActivation


class FPN(nn.Module):
    """
    FPN (Feature Pyramid Network) for multi-scale feature map extraction and merging.
    Uses 1x1 convolutions for output layers and 3x3 convolutions for merging layers.
    """

    def __init__(self, in_channels_list: list[int], out_channels: int) -> None:
        """
        Initializes the FPN module.

        Args:
            in_channels_list (list of int): List of input channel sizes for each pyramid level.
            out_channels (int): Number of output channels for the feature pyramid.
        """
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0
        # Define 1x1 convolution output layers
        self.output1 = Conv2dNormActivation(
            in_channels_list[0], out_channels, kernel_size=1, negative_slope=leaky
        )
        self.output2 = Conv2dNormActivation(
            in_channels_list[1], out_channels, kernel_size=1, negative_slope=leaky
        )
        self.output3 = Conv2dNormActivation(
            in_channels_list[2], out_channels, kernel_size=1, negative_slope=leaky
        )

        # Define merge layers using 3x3 convolutions
        self.merge1 = Conv2dNormActivation(
            out_channels, out_channels, kernel_size=3, negative_slope=leaky
        )
        self.merge2 = Conv2dNormActivation(
            out_channels, out_channels, kernel_size=3, negative_slope=leaky
        )

    def forward(self, inputs) -> list[Tensor]:
        """
        Forward pass of the FPN module.

        Args:
            inputs (dict or list): Input feature maps from different levels of the pyramid.

        Returns:
            list: List of merged output feature maps at different scales.
        """
        inputs = list(inputs.values())

        # Apply output layers to each feature map
        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])

        # Merge outputs with upsampling and addition
        upsample3 = F.interpolate(output3, size=output2.shape[2:], mode="nearest")
        output2 = self.merge2(output2 + upsample3)

        upsample2 = F.interpolate(output2, size=output1.shape[2:], mode="nearest")
        output1 = self.merge1(output1 + upsample2)

        # Return merged feature maps
        return [output1, output2, output3]
