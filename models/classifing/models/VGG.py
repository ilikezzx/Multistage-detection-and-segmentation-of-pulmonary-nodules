import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

from .model import Module,Flatten

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [32, 32, 'M', 64, 64, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(128*2*2*2, 1)
        self.active = nn.Sigmoid()

    def forward(self, x):
        out = self.features(x)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.classifier(out)
        out = self.active(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv3d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm3d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool3d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)