#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DBCSCAN 
@Author  ：zzx
@Date    ：2022/1/12 13:37 
"""
import numpy as np


def Iou(box1, box2, wh=False):
    if not wh:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = (box1[0] - box1[2] / 2.0), (box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = (box1[0] + box1[2] / 2.0), (box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = (box2[0] - box2[2] / 2.0), (box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = (box2[0] + box2[2] / 2.0), (box2[1] + box2[3] / 2.0)

    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)

    return iou, inter_area


def Giou(box1, box2, wh=True):
    # 分别是第一个矩形左右上下的坐标
    if not wh:
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
    else:
        x1, y1 = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
        x2, y2 = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0
        x3, y3 = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
        x4, y4 = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0

    iou, inter_area = Iou(box1, box2, wh)
    area_C = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
    area_1 = (x2 - x1) * (y2 - y1)
    area_2 = (x4 - x3) * (y4 - y3)
    sum_area = area_1 + area_2

    add_area = sum_area - inter_area  # 两矩形并集的面积
    end_area = (area_C - add_area) / area_C  # 闭包区域中不属于两个框的区域占闭包区域的比重
    giou = iou - end_area
    # 转换值域[-1, 1] to [0, 1]
    giou = (giou + 1) / 2.0
    # print(iou, giou)
    return giou, iou
