#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：LungNoduleDetector 
@Author  ：zzx
@Date    ：2022/2/21 14:24 
"""
# 多模块尺寸 ASPP、PPM

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBRModule(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, padding=0, dilation=1):
        super(CBRModule, self).__init__()
        self.conv = nn.Conv3d(input_channel, output_channel, kernel_size=kernel_size,
                              stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(output_channel)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv = CBRModule(in_channel, depth, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = CBRModule(in_channel, depth, 1)
        self.atrous_block6 = CBRModule(in_channel, depth, 3, padding=6, dilation=6)
        self.atrous_block12 = CBRModule(in_channel, depth, 3, padding=12, dilation=12)
        self.atrous_block18 = CBRModule(in_channel, depth, 3, padding=18, dilation=18)

        self.conv_1x1_output = CBRModule(depth * 5, depth, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='trilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        out = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)  # 这里N=4与原文一致
        self.conv1 = CBRModule(in_channels, inter_channels, 1, **kwargs)  # 四个1x1卷积用来减小channel为原来的1/N
        self.conv2 = CBRModule(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = CBRModule(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = CBRModule(in_channels, inter_channels, 1, **kwargs)
        self.out = CBRModule(in_channels * 2, out_channels, 1)  # 最后的1x1卷积缩小为原来的channel

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool3d(size)  # 自适应的平均池化，目标size分别为1x1,2x2,3x3,6x6
        return avgpool(x)

    def upsample(self, x, size):  # 上采样使用双线性插值
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)  # concat 四个池化的结果
        x = self.out(x)
        return x
