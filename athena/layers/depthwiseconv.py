from typing import Union, Tuple

import torch.nn as nn


class DepthwiseConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: int = 0,
        dilation: int = 1,
    ):
        """
        Implementation of the depth wise convolution.

        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
            kernel_size (Union[int, Tuple[int, int]]): The size of the kernel.
            padding (int, optional): The padding. Defaults to 0.
            dilation (int, optional): The dilation to use. Defaults to 1.
        """

        super(DepthwiseConv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            dilation=dilation,
        )
        self.point = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        return self.point(x)
