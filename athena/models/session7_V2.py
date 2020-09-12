import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        dropout_value = 0.1
        # Convolution Block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3),padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),padding=1, groups=32, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),padding=1, groups=32, bias=False),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),padding=1, groups=32, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        # Convolution Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),padding=1, groups=64, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),padding=1, groups=64, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),padding=1, groups=64, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        # Convolution Block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),padding=1, groups=128, bias=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),padding=1, groups=128, bias=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),padding=1, groups=128, bias=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        # Convolution Block 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),padding=1, groups=256, bias=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),padding=1, groups=256, bias=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),padding=1, groups=256, bias=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )
        
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8),
            # nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=True)
        )

        self.fc = nn.Linear(256,10)
  
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x,dim = -1)
