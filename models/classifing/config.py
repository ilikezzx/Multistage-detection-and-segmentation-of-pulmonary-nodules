"""
    存放 Classifier 的配置信息
"""

import argparse
import numpy as np
from torchvision import transforms

information = {
    'ResNet': {
        'weight': r"../models/classifing/weights/ResNet.pth"
    },
    'In-ResNet': {
        'weight': r'../models/classifing/weights/In-ResNet.pth'
    },
    'VGG': {
        'weight': r'../models/classifing/weights/VGG.pth'
    },
    'ResXNet': {
        'weight': r'../models/classifing/weights/classifier/cp_resxnet_24@3/best.pth'
    },
    'ResNet_basic111_maxpool': {
        'weight': r'../models/classifing/weights/classifier/cp_resnet_64_64_64/best.pth'
    },
    'ResNet_small_basic111_maxpool':{
        'weight': r'../models/classifing/weights/classifier/cp_resnet_24@3/best.pth'
    }
}

transform = transforms.Compose([transforms.ToTensor()])
batch_size = 8
