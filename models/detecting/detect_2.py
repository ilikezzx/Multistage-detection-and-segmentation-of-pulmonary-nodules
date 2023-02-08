"""
    这一部分是输入切片，获得切片检测结果
"""
import sys
import argparse
import time
from pathlib import Path

import cv2
import yaml
import torch
import numpy as np
from PIL import Image
from numpy import random
from torch.utils.data import DataLoader

from .dataset import DetectorDataset
from ..config import load_locating_config, locating_information
from .models.experimental import attempt_load
from .models.yolo import Model
from .utils.datasets import LoadStreams, LoadImages
from .utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from .utils.plots import plot_one_box
from .utils.torch_utils import select_device, load_classifier, time_synchronized, intersect_dicts


def detect_position_2(imgs, model_name=None, pb=None):
    """
    :param pb: 进度条
    :param imgs: 切片集，数值范围[0.0~1.0]
    :return:
        a list of dict
            -- index
            -- new_image
            -- positions
    """
    opt = load_locating_config()
    weights, imgsz = opt.weights, opt.img_size

    if model_name is not None:
        opt.cfg = locating_information[model_name]['cfg']
        weights = locating_information[model_name]['weights']

    # Initialize
    # set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # print(weights, opt.cfg)

    # Load model
    model = Model(opt.cfg, ch=3, nc=1)
    state_dict = torch.load(weights, map_location=device)  # to FP32
    # state_dict = intersect_dicts(state_dict, model.state_dict())  # intersect
    print(state_dict.keys())

    model.load_state_dict(state_dict)  # load
    model.to(device).eval()
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    valid_images = []

    dataset = DetectorDataset(imgs)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    t0 = time.time()
    for index, batchs in enumerate(dataloader):
        images, _ = batchs
        img0_size = (640, 640, 3)

        # print(images.shape)
        with torch.no_grad():
            pred = model(images, augment=opt.augment)[0]
            pred = non_max_suppression(pred, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)
            gn = torch.tensor(img0_size)[[1, 0, 1, 0]]  # normalization gain whwh

            # print(len(pred))
            # print(pred)
            # print(pred[-1])
            # pred = pred[-1]  # 因为维度是1
            for slice_cnt, slice_pred in enumerate(pred):
                if len(slice_pred):
                    slice_pred[:, :4] = scale_coords(images.shape[2:], slice_pred[:, :4], img0_size).round()

                    for *xyxy, conf, cls in reversed(slice_pred):
                        valid_image = {}
                        positions = []
                        valid_image.setdefault('index', opt.batch_size * index + slice_cnt)

                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        positions.append(xywh)
                        valid_image.setdefault('positions', positions)
                        valid_image.setdefault('conf', conf)

                        valid_images.append(valid_image)
    t1 = time.time()
    print(f'detect time:{t1 - t0}, slice_num: {imgs.shape[0]},every slice average={(t1 - t0) / imgs.shape[0]}')
    return valid_images


if __name__ == '__main__':
    image = np.load('1.npy')
    detect_position_2(image)
