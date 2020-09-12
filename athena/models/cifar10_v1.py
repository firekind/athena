import torch.nn as nn
import torch.nn.functional as F

from athena.layers import DepthwiseConv2d


class Cifar10V1(nn.Module):
    def __init__(self, in_channels: int=3, dropout_value: float=0.25):
        """
        Cifar 10 model made for assignment 7. Implemented by Shyamant Achar.

        Args:
            in_channels (int, optional): The number of input channels. Defaults to 3.
            dropout_value (float, optional): The dropout percentage. Defaults to 0.25.
        """

        super(Cifar10V1, self).__init__()

        self.block1 = nn.Sequential(
            DepthwiseConv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            DepthwiseConv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            DepthwiseConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        self.dilationblock1 = nn.Sequential(
            DepthwiseConv2d(3, 32, 3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            DepthwiseConv2d(32, 64, 3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            DepthwiseConv2d(64, 128, 3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 32, 1),
        )

        self.block2 = nn.Sequential(
            DepthwiseConv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            DepthwiseConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            DepthwiseConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        self.dilationblock2 = nn.Sequential(
            DepthwiseConv2d(32, 64, 3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            DepthwiseConv2d(64, 128, 3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            DepthwiseConv2d(128, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        self.transition2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 64, 1),
        )

        self.block3 = nn.Sequential(
            DepthwiseConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            DepthwiseConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            DepthwiseConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        self.dilationblock3 = nn.Sequential(
            DepthwiseConv2d(64, 128, 3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            DepthwiseConv2d(128, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            DepthwiseConv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        self.out_block = nn.Sequential(
            nn.AvgPool2d(8),
            nn.Conv2d(256, 64, 1),
            nn.Conv2d(64, 10, 1),
        )

    def forward(self, x):
        dep_x = self.block1(x)
        dia_x = self.dilationblock1(x)
        x = dep_x + dia_x
        x = self.transition1(x)

        dep_x = self.block2(x)
        dia_x = self.dilationblock2(x)
        x = dep_x + dia_x
        x = self.transition2(x)

        dep_x = self.block3(x)
        dia_x = self.dilationblock3(x)
        x = dep_x + dia_x
        x = self.out_block(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)