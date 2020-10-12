import torch
import torch.nn as nn

from .resnet import BasicBlock


class LayerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LayerBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.resblock = BasicBlock(out_channels, out_channels)

    def forward(self, x):
        out = self.block(x)
        res_out = self.resblock(out)

        return res_out + out


class DavidNet(nn.Module):
    def __init__(self):
        """
        A Simple implementation of DavidNet_.

        .. _DavidNet: https://github.com/stanford-futuredata/dawn-bench-entries/blob/master/CIFAR10/train/davidcpage_resnet9_1v100-ec2_pytorch.json
        """
        super(DavidNet, self).__init__()

        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )

        self.layer1 = LayerBlock(64, 128)

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer3 = LayerBlock(256, 512)

        self.maxpool = nn.MaxPool2d(4, 1)
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.linear(x.view(x.size(0), -1))

        return x
