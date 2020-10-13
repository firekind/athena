import torch.nn as nn
import torch.nn.functional as F

from athena.layers import DepthwiseConv2d


class Cifar10V1(nn.Module):
    def __init__(self, in_channels: int = 3, dropout_value: float = 0.25):
        """
        Cifar 10 model made for assignment 7. Implemented by Shyamant Achar. Model summary:

        .. code-block:: none

            ----------------------------------------------------------------
                    Layer (type)               Output Shape         Param #
            ================================================================
                        Conv2d-1            [-1, 3, 32, 32]              30
                        Conv2d-2           [-1, 32, 32, 32]             128
               DepthwiseConv2d-3           [-1, 32, 32, 32]               0
                   BatchNorm2d-4           [-1, 32, 32, 32]              64
                          ReLU-5           [-1, 32, 32, 32]               0
                       Dropout-6           [-1, 32, 32, 32]               0
                        Conv2d-7           [-1, 32, 32, 32]             320
                        Conv2d-8           [-1, 64, 32, 32]           2,112
               DepthwiseConv2d-9           [-1, 64, 32, 32]               0
                  BatchNorm2d-10           [-1, 64, 32, 32]             128
                         ReLU-11           [-1, 64, 32, 32]               0
                      Dropout-12           [-1, 64, 32, 32]               0
                       Conv2d-13           [-1, 64, 32, 32]             640
                       Conv2d-14          [-1, 128, 32, 32]           8,320
              DepthwiseConv2d-15          [-1, 128, 32, 32]               0
                  BatchNorm2d-16          [-1, 128, 32, 32]             256
                         ReLU-17          [-1, 128, 32, 32]               0
                      Dropout-18          [-1, 128, 32, 32]               0
                       Conv2d-19            [-1, 3, 32, 32]              30
                       Conv2d-20           [-1, 32, 32, 32]             128
              DepthwiseConv2d-21           [-1, 32, 32, 32]               0
                  BatchNorm2d-22           [-1, 32, 32, 32]              64
                         ReLU-23           [-1, 32, 32, 32]               0
                      Dropout-24           [-1, 32, 32, 32]               0
                       Conv2d-25           [-1, 32, 32, 32]             320
                       Conv2d-26           [-1, 64, 32, 32]           2,112
              DepthwiseConv2d-27           [-1, 64, 32, 32]               0
                  BatchNorm2d-28           [-1, 64, 32, 32]             128
                         ReLU-29           [-1, 64, 32, 32]               0
                      Dropout-30           [-1, 64, 32, 32]               0
                       Conv2d-31           [-1, 64, 32, 32]             640
                       Conv2d-32          [-1, 128, 32, 32]           8,320
              DepthwiseConv2d-33          [-1, 128, 32, 32]               0
                  BatchNorm2d-34          [-1, 128, 32, 32]             256
                         ReLU-35          [-1, 128, 32, 32]               0
                      Dropout-36          [-1, 128, 32, 32]               0
                    MaxPool2d-37          [-1, 128, 16, 16]               0
                       Conv2d-38           [-1, 32, 16, 16]           4,128
                       Conv2d-39           [-1, 32, 16, 16]             320
                       Conv2d-40           [-1, 64, 16, 16]           2,112
              DepthwiseConv2d-41           [-1, 64, 16, 16]               0
                  BatchNorm2d-42           [-1, 64, 16, 16]             128
                         ReLU-43           [-1, 64, 16, 16]               0
                      Dropout-44           [-1, 64, 16, 16]               0
                       Conv2d-45           [-1, 64, 16, 16]             640
                       Conv2d-46          [-1, 128, 16, 16]           8,320
              DepthwiseConv2d-47          [-1, 128, 16, 16]               0
                  BatchNorm2d-48          [-1, 128, 16, 16]             256
                         ReLU-49          [-1, 128, 16, 16]               0
                      Dropout-50          [-1, 128, 16, 16]               0
                       Conv2d-51          [-1, 128, 16, 16]           1,280
                       Conv2d-52          [-1, 256, 16, 16]          33,024
              DepthwiseConv2d-53          [-1, 256, 16, 16]               0
                  BatchNorm2d-54          [-1, 256, 16, 16]             512
                         ReLU-55          [-1, 256, 16, 16]               0
                      Dropout-56          [-1, 256, 16, 16]               0
                       Conv2d-57           [-1, 32, 16, 16]             320
                       Conv2d-58           [-1, 64, 16, 16]           2,112
              DepthwiseConv2d-59           [-1, 64, 16, 16]               0
                  BatchNorm2d-60           [-1, 64, 16, 16]             128
                         ReLU-61           [-1, 64, 16, 16]               0
                      Dropout-62           [-1, 64, 16, 16]               0
                       Conv2d-63           [-1, 64, 16, 16]             640
                       Conv2d-64          [-1, 128, 16, 16]           8,320
              DepthwiseConv2d-65          [-1, 128, 16, 16]               0
                  BatchNorm2d-66          [-1, 128, 16, 16]             256
                         ReLU-67          [-1, 128, 16, 16]               0
                      Dropout-68          [-1, 128, 16, 16]               0
                       Conv2d-69          [-1, 128, 16, 16]           1,280
                       Conv2d-70          [-1, 256, 16, 16]          33,024
              DepthwiseConv2d-71          [-1, 256, 16, 16]               0
                  BatchNorm2d-72          [-1, 256, 16, 16]             512
                         ReLU-73          [-1, 256, 16, 16]               0
                      Dropout-74          [-1, 256, 16, 16]               0
                    MaxPool2d-75            [-1, 256, 8, 8]               0
                       Conv2d-76             [-1, 64, 8, 8]          16,448
                       Conv2d-77             [-1, 64, 8, 8]             640
                       Conv2d-78            [-1, 128, 8, 8]           8,320
              DepthwiseConv2d-79            [-1, 128, 8, 8]               0
                  BatchNorm2d-80            [-1, 128, 8, 8]             256
                         ReLU-81            [-1, 128, 8, 8]               0
                      Dropout-82            [-1, 128, 8, 8]               0
                       Conv2d-83            [-1, 128, 8, 8]           1,280
                       Conv2d-84            [-1, 256, 8, 8]          33,024
              DepthwiseConv2d-85            [-1, 256, 8, 8]               0
                  BatchNorm2d-86            [-1, 256, 8, 8]             512
                         ReLU-87            [-1, 256, 8, 8]               0
                      Dropout-88            [-1, 256, 8, 8]               0
                       Conv2d-89            [-1, 256, 8, 8]           2,560
                       Conv2d-90            [-1, 256, 8, 8]          65,792
              DepthwiseConv2d-91            [-1, 256, 8, 8]               0
                  BatchNorm2d-92            [-1, 256, 8, 8]             512
                         ReLU-93            [-1, 256, 8, 8]               0
                      Dropout-94            [-1, 256, 8, 8]               0
                       Conv2d-95             [-1, 64, 8, 8]             640
                       Conv2d-96            [-1, 128, 8, 8]           8,320
              DepthwiseConv2d-97            [-1, 128, 8, 8]               0
                  BatchNorm2d-98            [-1, 128, 8, 8]             256
                         ReLU-99            [-1, 128, 8, 8]               0
                     Dropout-100            [-1, 128, 8, 8]               0
                      Conv2d-101            [-1, 128, 8, 8]           1,280
                      Conv2d-102            [-1, 256, 8, 8]          33,024
             DepthwiseConv2d-103            [-1, 256, 8, 8]               0
                 BatchNorm2d-104            [-1, 256, 8, 8]             512
                        ReLU-105            [-1, 256, 8, 8]               0
                     Dropout-106            [-1, 256, 8, 8]               0
                      Conv2d-107            [-1, 256, 8, 8]           2,560
                      Conv2d-108            [-1, 256, 8, 8]          65,792
             DepthwiseConv2d-109            [-1, 256, 8, 8]               0
                 BatchNorm2d-110            [-1, 256, 8, 8]             512
                        ReLU-111            [-1, 256, 8, 8]               0
                     Dropout-112            [-1, 256, 8, 8]               0
                   AvgPool2d-113            [-1, 256, 1, 1]               0
                      Conv2d-114             [-1, 64, 1, 1]          16,448
                      Conv2d-115             [-1, 10, 1, 1]             650
            ================================================================
            Total params: 380,646
            Trainable params: 380,646
            Non-trainable params: 0
            ----------------------------------------------------------------
            Input size (MB): 0.01
            Forward/backward pass size (MB): 32.71
            Params size (MB): 1.45
            Estimated Total Size (MB): 34.17
            ----------------------------------------------------------------

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
