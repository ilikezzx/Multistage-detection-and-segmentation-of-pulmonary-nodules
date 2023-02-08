#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：main.py 
@Author  ：zzx
@Date    ：2022/1/15 19:16 
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .model import Module, Flatten


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality=32, base_width=4, widen_factor=4):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)

        self.conv_reduce = nn.Conv3d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm3d(D)
        self.conv_conv = nn.Conv3d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm3d(D)
        self.conv_expand = nn.Conv3d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm3d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        # print(residual.shape, bottleneck.shape)
        return F.relu(residual + bottleneck, inplace=True)


class ResXNet(Module):
    def __init__(self, in_channel: int, out_channels: list, class_num: int = 2):
        super(ResXNet, self).__init__()
        self.model_name = 'ResXNet'
        assert out_channels is not None and len(out_channels) >= 4, "Please check out_channels"
        self.stage_1 = ResNeXtBottleneck(in_channel, out_channels[0], stride=1)  # bs*out_channels[0]*24^3
        self.stage_2 = ResNeXtBottleneck(out_channels[0], out_channels[1], stride=2)  # bs*out_channels[1]*12^3
        self.stage_3 = ResNeXtBottleneck(out_channels[1], out_channels[2], stride=2)  # bs*out_channels[2]*6^3
        self.stage_4 = ResNeXtBottleneck(out_channels[2], out_channels[3], stride=2)  # bs*out_channels[3]*3^3
        #         self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # bs*out_channels[3]*1*1*1
        self.flatten = Flatten()
        self.fc1 = nn.Linear(out_channels[3] * 3 * 3 * 3, class_num)
        self.active = nn.Sigmoid()

    def forward(self, x1):
        out1 = self.stage_1(x1)
        out1 = self.stage_2(out1)
        out1 = self.stage_3(out1)
        out1 = self.stage_4(out1)
        out1 = self.flatten(out1)
        out1 = self.fc1(out1)
        out1 = self.active(out1)

        return out1

class ResXNet_2(Module):
    def __init__(self, in_channel: int, out_channels: list, class_num: int = 2):
        super(ResXNet_2, self).__init__()
        self.model_name = 'ResXNet'
        assert out_channels is not None and len(out_channels) >= 4, "Please check out_channels"
        self.stage_1 = ResNeXtBottleneck(in_channel, out_channels[0], stride=1)  # bs*out_channels[0]*24^3
        self.stage_2 = ResNeXtBottleneck(out_channels[0], out_channels[1], stride=2)  # bs*out_channels[1]*12^3
        self.stage_3 = ResNeXtBottleneck(out_channels[1], out_channels[2], stride=2)  # bs*out_channels[2]*6^3
        self.stage_4 = ResNeXtBottleneck(out_channels[2], out_channels[3], stride=2)  # bs*out_channels[3]*3^3
        #         self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # bs*out_channels[3]*1*1*1
        self.flatten = Flatten()
        self.fc1 = nn.Linear(out_channels[3] * 8 * 3 * 8, class_num)
        self.active = nn.Sigmoid()

    def forward(self, x1):
        out1 = self.stage_1(x1)
        out1 = self.stage_2(out1)
        out1 = self.stage_3(out1)
        out1 = self.stage_4(out1)
        out1 = self.flatten(out1)
        out1 = self.fc1(out1)
        out1 = self.active(out1)

        return out1


if __name__ == '__main__':
    images = np.random.rand(32, 32, 32)
    # print(images.shape)
    images = torch.tensor(images, dtype=torch.float32)
    images = images.unsqueeze(0)
    images = images.unsqueeze(0)
    model = ResXNet(1, [16 * 4, 32 * 4, 64 * 4, 128 * 4])
    pred = model(images)
