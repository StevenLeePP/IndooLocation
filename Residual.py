import torch
from torch import nn
from collections import OrderedDict
from torch.nn import functional as F


class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=(1,1)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=(3,1), padding=(1,0), stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=(3,1), padding=(1,0))
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,use_1x1conv=True, strides=(4,2)))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

def Student():
    b1 = nn.Sequential(OrderedDict([
                    ('Conv',nn.Conv2d(1, 8, kernel_size=(3,2),stride = (1,2),padding = (1,0))),    # 之前是1
                    ('BatchNorm2d',nn.BatchNorm2d(8)),
                    ('RelU',nn.ReLU()),
                    ('MaxPool',nn.MaxPool2d(kernel_size=(5,1), stride=(2,1),padding = (2,0)))
                    ]))
    b2 = nn.Sequential(*resnet_block(8, 16, 2))
    b3 = nn.Sequential(*resnet_block(16, 32, 2))
    # b4 = nn.Sequential(*resnet_block(64, 128, 2))
    # b5 = nn.Sequential(*resnet_block(128, 256, 2))
    net = nn.Sequential(b1, b2,b3,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(32, 2))

    return net