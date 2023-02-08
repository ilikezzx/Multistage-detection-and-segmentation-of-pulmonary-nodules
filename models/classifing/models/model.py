import os
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .resblock import BasicBlock, Bottleneck
from .Inception import InceptionBlock


class Module(nn.Module):
    """分类器模块抽象类"""

    def __init__(self):
        super(Module, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        state = torch.load(path)
        self.load_state_dict(state['model_state_dict'])

    def save(self, name=None):
        if name is None:
            suffix = 'checkpoints' + os.sep + self.model_name + "_"
            name = time.strftime(suffix + "%m%d_%H:%M:%S.pth")
        torch.save(self.state_dict(), name)


class Flatten(nn.Module):
    """将切块拍平"""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


# 统一版Model
class ResNet(Module):
    def __init__(self, in_channel: int, every_layers_message: list, layer_type=BasicBlock,
                 downsample_type=nn.MaxPool3d, class_num: int = 1):
        """
        :param in_channel:   输入图像的通道
        :param every_layers_message:
            a list of [(layer_num, out_channel),(...),...,]
                --  layer_num: 特征提取，数值越大，模型越深
                --  out_channel: 每个block的输出通道数
        :param class_num: 分类数，二分类数值 == 1
        """
        super(ResNet, self).__init__()
        if class_num == 2:
            class_num = 1
        # isBasic = 1 if layer_type == BasicBlock else 4

        assert len(every_layers_message) >= 3, "please check layers_num"
        last_out_channel = None
        self.blocks = []
        for index, (layer_num, out_channel) in enumerate(every_layers_message):
            # appending extract features layer
            for _ in range(layer_num):
                self.blocks.append(layer_type(in_channel, out_channel))
                # print(in_channel, out_channel)
                in_channel = out_channel

            last_out_channel = out_channel

            # appending down sample layer
            if downsample_type == nn.MaxPool3d:
                self.blocks.append(downsample_type(kernel_size=3, stride=2, padding=1))
            else:
                self.blocks.append(downsample_type(out_channel, out_channel))

        self.blocks = nn.ModuleList(self.blocks)
        self.ef = layer_type(last_out_channel, last_out_channel)
        # last_layer
        self.downsample = nn.Sequential(nn.Conv3d(last_out_channel, last_out_channel, (1, 1, 1), padding=0),
                                        nn.BatchNorm3d(last_out_channel),
                                        nn.ReLU(True),
                                        Flatten())

        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 4 * last_out_channel, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, class_num)
        )
        # active
        self.active = None
        if class_num == 1:
            self.active = nn.Sigmoid()
        else:
            self.active = nn.Softmax()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        out = self.downsample(self.ef(x))
        # print(out.shape)
        out = self.fc(out)
        out = self.active(out)

        return out


# 统一版Model
class ResNet_small(Module):
    def __init__(self, in_channel: int, every_layers_message: list, layer_type=BasicBlock,
                 downsample_type=nn.MaxPool3d, class_num: int = 1):
        """
        :param in_channel:   输入图像的通道
        :param every_layers_message:
            a list of [(layer_num, out_channel),(...),...,]
                --  layer_num: 特征提取，数值越大，模型越深
                --  out_channel: 每个block的输出通道数
        :param class_num: 分类数，二分类数值 == 1
        """
        super(ResNet_small, self).__init__()
        if class_num == 2:
            class_num = 1
        # isBasic = 1 if layer_type == BasicBlock else 4

        assert len(every_layers_message) == 3, "please check layers_num"
        last_out_channel = None
        self.blocks = []
        for index, (layer_num, out_channel) in enumerate(every_layers_message):
            # appending extract features layer
            for _ in range(layer_num):
                self.blocks.append(layer_type(in_channel, out_channel))
                # print(in_channel, out_channel)
                in_channel = out_channel

            last_out_channel = out_channel

            # appending down sample layer
            if downsample_type == nn.MaxPool3d:
                self.blocks.append(downsample_type(kernel_size=3, stride=2, padding=1))
            else:
                self.blocks.append(downsample_type(out_channel, out_channel))

        self.blocks = nn.ModuleList(self.blocks)
        self.ef = layer_type(last_out_channel, last_out_channel)
        # last_layer
        self.downsample = nn.Sequential(nn.Conv3d(last_out_channel, last_out_channel, (1, 1, 1), padding=0),
                                        nn.BatchNorm3d(last_out_channel),
                                        nn.ReLU(True),
                                        Flatten())

        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 3 * last_out_channel, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, class_num)
        )
        # active
        self.active = None
        if class_num == 1:
            self.active = nn.Sigmoid()
        else:
            self.active = nn.Softmax()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        out = self.downsample(self.ef(x))
        # print(out.shape)
        out = self.fc(out)
        out = self.active(out)

        return out


class ResNet_small_basic111_maxpool(ResNet_small):
    def __init__(self, in_channel: int, every_layers_message: list, layer_type=BasicBlock,
                 downsample_type=nn.MaxPool3d, class_num: int = 1):
        super(ResNet_small_basic111_maxpool, self).__init__(in_channel, every_layers_message, layer_type=BasicBlock,
                                                            downsample_type=nn.MaxPool3d, class_num=class_num)


class ResNet_basic111_maxpool(ResNet):
    def __init__(self, in_channel: int, every_layers_message: list, layer_type=BasicBlock,
                 downsample_type=nn.MaxPool3d, class_num: int = 1):
        super(ResNet_basic111_maxpool, self).__init__(in_channel, every_layers_message, layer_type=BasicBlock,
                                                      downsample_type=nn.MaxPool3d, class_num=class_num)


class ResNet_basic111_inception(ResNet):
    def __init__(self, in_channel: int, every_layers_message: list, layer_type=BasicBlock,
                 downsample_type=InceptionBlock, class_num: int = 1):
        super(ResNet_basic111_inception, self).__init__(in_channel, every_layers_message, layer_type=layer_type,
                                                        downsample_type=downsample_type, class_num=class_num)


class ResNet_bl111_inception(ResNet):
    def __init__(self, in_channel: int, every_layers_message: list, layer_type=Bottleneck,
                 downsample_type=InceptionBlock, class_num: int = 1):
        super(ResNet_bl111_inception, self).__init__(in_channel, every_layers_message, layer_type=layer_type,
                                                     downsample_type=downsample_type, class_num=class_num)



if __name__ == '__main__':
    # images = np.random.rand(64, 64, 64)
    # # print(images.shape)
    # images = torch.tensor(images, dtype=torch.float32)
    # images = images.unsqueeze(0)
    # images = images.unsqueeze(0)
    # model = Classifier_2(1, [16, 32, 64, 64])
    # pred = model(images)

    sum1 = sum(x.numel() for x in Classifier_1(1, [32, 64, 64, 64]).parameters())
    sum2 = sum(x.numel() for x in Classifier_2(1, [32, 64, 64, 64]).parameters())

    print(sum1, sum2)
