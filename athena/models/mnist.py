import torch
import torch.nn as nn
import torch.nn.functional as F
from athena.layers import GhostBatchNorm


class MnistNet(nn.Module):
    def __init__(
        self,
        use_ghost_batch_norm: bool = False,
        num_splits: int = 2,
        dropout_value: float = 0,
    ):
        """
        Model used on MNIST dataset. Model summary:

        .. code-block:: none

            ----------------------------------------------------------------
                    Layer (type)               Output Shape         Param #
            ================================================================
                        Conv2d-1            [-1, 7, 26, 26]              63
                   BatchNorm2d-2            [-1, 7, 26, 26]              14
                          ReLU-3            [-1, 7, 26, 26]               0
                       Dropout-4            [-1, 7, 26, 26]               0
                        Conv2d-5           [-1, 16, 24, 24]           1,008
                   BatchNorm2d-6           [-1, 16, 24, 24]              32
                          ReLU-7           [-1, 16, 24, 24]               0
                       Dropout-8           [-1, 16, 24, 24]               0
                        Conv2d-9           [-1, 12, 24, 24]             192
                    MaxPool2d-10           [-1, 12, 12, 12]               0
                       Conv2d-11           [-1, 16, 10, 10]           1,728
                  BatchNorm2d-12           [-1, 16, 10, 10]              32
                         ReLU-13           [-1, 16, 10, 10]               0
                      Dropout-14           [-1, 16, 10, 10]               0
                       Conv2d-15             [-1, 16, 8, 8]           2,304
                  BatchNorm2d-16             [-1, 16, 8, 8]              32
                         ReLU-17             [-1, 16, 8, 8]               0
                      Dropout-18             [-1, 16, 8, 8]               0
                       Conv2d-19             [-1, 16, 6, 6]           2,304
                  BatchNorm2d-20             [-1, 16, 6, 6]              32
                         ReLU-21             [-1, 16, 6, 6]               0
                      Dropout-22             [-1, 16, 6, 6]               0
                    AvgPool2d-23             [-1, 16, 1, 1]               0
                       Conv2d-24             [-1, 10, 1, 1]             170
            ================================================================
            Total params: 7,911
            Trainable params: 7,911
            Non-trainable params: 0
            ----------------------------------------------------------------
            Input size (MB): 0.00
            Forward/backward pass size (MB): 0.59
            Params size (MB): 0.03
            Estimated Total Size (MB): 0.62
            ----------------------------------------------------------------


        Args:
            use_ghost_batch_norm (bool, optional): Whether to use Ghost Batch Norm instead of regular Batch Norm. Defaults to False.
            num_splits (int, optional): The number of splits to use in Ghost Batch Norm. Defaults to 2.
            dropout_value (float, optional): The percentage of dropout. Defaults to 0.
        """

        super(MnistNet, self).__init__()
        # Convolution Block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=7, kernel_size=(3, 3), padding=0, bias=False
            ),
            (
                nn.BatchNorm2d(7)
                if not use_ghost_batch_norm
                else GhostBatchNorm(7, num_splits)
            ),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        # Convolution Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=7,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            (
                nn.BatchNorm2d(16)
                if not use_ghost_batch_norm
                else GhostBatchNorm(16, num_splits)
            ),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        # Transition Block 1
        self.transitionblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=12,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
            nn.MaxPool2d(2, 2),
        )

        # Convolution Block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            (
                nn.BatchNorm2d(16)
                if not use_ghost_batch_norm
                else GhostBatchNorm(16, num_splits)
            ),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        # Convolution Block 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            (
                nn.BatchNorm2d(16)
                if not use_ghost_batch_norm
                else GhostBatchNorm(16, num_splits)
            ),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            (
                nn.BatchNorm2d(16)
                if not use_ghost_batch_norm
                else GhostBatchNorm(16, num_splits)
            ),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5),
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the model.
        """

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transitionblock1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return x
