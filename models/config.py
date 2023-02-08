#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import numpy as np
from torchvision import transforms

##########################################
# first stage: locating
##########################################

locating_information = {
    'Yolo': {
        'weights': r'../models/detecting/weights/yolo_2d.pt',
        'cfg': r'../models/detecting/data/model.yaml'
    }
}


def load_locating_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'./models/detecting/weights/yolov5l.pt',
                        help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='./models/detecting/data/yolov5l.yaml',
                        help='hyperparameters path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--batch-size', type=int, default=16, help='batch value')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    # print(opt)
    return opt


##########################################
# Third stage: classification
##########################################
classification_information = {
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
    'ResNet_small_basic111_maxpool': {
        'weight': r'../models/classifing/weights/classifier/cp_resnet_24@3/best.pth'
    }
}

classification_transform = transforms.Compose([transforms.ToTensor()])
classification_batch_size = 8

##########################################
# forth stage: segmentation
##########################################
segmentation_information = {'Base': r"../models/segmenting/weights/base.pt",
                            'Res': r"../models/segmenting/weights/res.pt"}


def load_segmentation_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Base')
    parser.add_argument('--weights', nargs='+', type=str, default=r'../models/segmenting/weights/base.pt',
                        help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--out-channels', type=int, default=1)
    parser.add_argument('--f-maps', type=int, default=32)

    opt = parser.parse_args()
    # print(opt)
    return opt
