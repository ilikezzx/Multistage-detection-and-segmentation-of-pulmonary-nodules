"""
    3D-ResNet
"""

import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


def conv3x3x3(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, padding: int = 1,
              groups: int = 1) -> nn.Conv3d:
    """3x3x3 convolution padding"""
    return nn.Conv3d(in_channels, out_channels,
                     kernel_size=3, stride=stride,
                     bias=False, padding=padding,
                     groups=groups, dilation=dilation)


def conv1x1x1(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, padding: int = 1,
              groups: int = 1) -> nn.Conv3d:
    """1x1x1 convolution padding"""
    return nn.Conv3d(in_channels, out_channels,
                     kernel_size=1, stride=stride,
                     bias=False, padding=padding,
                     groups=groups, dilation=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 padding:int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self.downsample = downsample
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding

        self.conv1 = conv3x3x3(in_channels, out_channels, stride=stride, padding=padding)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = conv3x3x3(out_channels, out_channels, stride=stride, padding=padding)
        self.bn2 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if self.downsample is None:
            self.downsample = nn.Sequential(
                conv1x1x1(in_channels, out_channels, stride=stride, padding=0),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 padding: int = 1,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self.downsample = downsample
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding

        self.conv1 = conv1x1x1(in_channels, out_channels, stride=self.stride, dilation=dilation, padding=padding)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = conv3x3x3(out_channels, out_channels, stride=self.stride, dilation=dilation, padding=padding)
        self.bn2 = norm_layer(out_channels)
        self.conv3 = conv1x1x1(out_channels, out_channels * self.expansion, stride=self.stride, dilation=dilation,
                               padding=padding)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # connection
        out += residual
        out = self.relu(out)
        return out
