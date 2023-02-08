"""
    根据论文改写
"""

import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """
        输入尺寸:
            [batch_size,in_channel,depth,height,width]
        输出尺寸：
            [batch_size,out_channel,depth//2,height//2,width//2]
    """

    def __init__(self, in_channel, out_channel, relu=True, bn=True):
        super(InceptionBlock, self).__init__()
        assert out_channel % 4 == 0, "please check Inception Outputs Channels!"

        out_channels = [out_channel // 4] * 4
        self.active = nn.Sequential()
        if bn:
            self.active.add_module('batch_norm', nn.BatchNorm3d(out_channel))
        if relu:
            self.active.add_module('relu', nn.ReLU(inplace=True))

        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channel, out_channels[0], kernel_size=1, stride=2)
        )

        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channel, 2 * out_channels[1], kernel_size=1),
            nn.BatchNorm3d(2 * out_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(2 * out_channels[1], out_channels[1], kernel_size=3, stride=2, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channel, 2 * out_channels[2], kernel_size=1, stride=1),
            nn.BatchNorm3d(2 * out_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv3d(2 * out_channels[2], out_channels[2], 5, stride=2, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(in_channel, out_channels[3], kernel_size=1, stride=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat((branch1, branch2, branch3, branch4), 1)
        out = self.active(out)
        return out
