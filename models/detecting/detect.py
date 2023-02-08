"""
    这一部分是输入切片，获得切片检测结果
"""

import argparse
import time
from pathlib import Path

import cv2
import yaml
import torch
import numpy as np
from PIL import Image
from numpy import random

from .config import load_config,information
from .models.experimental import attempt_load
from .models.yolo import Model
from .utils.datasets import LoadStreams, LoadImages
from .utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from .utils.plots import plot_one_box
from .utils.torch_utils import select_device, load_classifier, time_synchronized, intersect_dicts


def detect_position(imgs, model_name=None,pb=None):
    """
    :param pb: 进度条
    :param imgs: 切片集，数值范围[0.0~1.0]
    :return:
        a list of dict
            -- index
            -- new_image
            -- positions
    """

    opt = load_config()
    weights, imgsz = opt.weights, opt.img_size

    if model_name is not None:
        opt.cfg = information[model_name]['cfg']
        weights = information[model_name]['weights']

    # print(weights)

    # Initialize
    # set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # ckpt = torch.load(weights, map_location=device)
    model = Model(opt.cfg, ch=3, nc=1)
    state_dict = torch.load(weights)  # to FP32
    # state_dict = intersect_dicts(state_dict, model.state_dict())  # intersect
    model.load_state_dict(state_dict, strict=False)  # load
    model.to(device).eval()
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = ['NSCLC']
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors = [[255, 0, 0]]

    t0 = time.time()
    valid_imgs = []
    for index, img in enumerate(imgs):
        # 改变图像尺寸
        img = Image.fromarray(img).resize((640, 640))
        img = np.array(img, dtype=np.float)

        img = np.stack((img,) * 3, axis=-1)
        # copy image
        im0 = img.copy() * 255.0
        img = torch.from_numpy(img).to(device, non_blocking=True).float()
        img = img.half() if half else img.float()  # uint8==>fp 16/32

        # img dimension = batch_size,channels,height,width
        img = img.permute(2, 0, 1)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
            # apply nms
            pred = non_max_suppression(pred, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)
            t2 = time_synchronized()

            # s += '%gx%g ' % img.shape[2:]  # print string
            # 因为每次添加输入一个切片,即batch_size为1
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            pred = pred[-1]  # 因为维度是1
            if len(pred):
                pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(pred):
                    valid_img = {}
                    positions = []
                    temp_image = np.copy(im0)
                    label = '%s %.2f' % (names[int(cls)], conf)  # name+置信度
                    plot_one_box(xyxy, temp_image, label=label, color=colors[int(cls)], line_thickness=4)
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    positions.append(xywh)

                    # cv2.imwrite('save.png', im0)
                    valid_img.setdefault('index', index)
                    valid_img.setdefault('new_image', temp_image)
                    valid_img.setdefault('positions', positions)
                    valid_img.setdefault('conf', conf)
                    valid_imgs.append(valid_img)
                    print(round(positions[0][0]*512.0), round(positions[0][1]*512.0), round(positions[0][2]*512.0),
                          round(positions[0][3]*512.0), index)

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))
        if index % max(round(len(imgs) * 3 / 20.0), 1) == 0 and pb is not None:
            pb.update(index / len(imgs), 1, 3)
    return valid_imgs


if __name__ == '__main__':
    image = np.load('1.npy')
    detect_position(image)
