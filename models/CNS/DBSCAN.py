#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DBCSCAN 
@Author  ：zzx
@Date    ：2022/1/12 11:20 
"""
"""
    SNN算法，kNN和层次聚类的结合体
"""
import cv2
import copy
import math
import numpy as np

from queue import PriorityQueue
from .iou import Giou
from .cluster import Cluster


def HC(data, threshold=10.0):
    que = PriorityQueue()  # 优先队列
    dis_lists = []  # 存放距离
    valid_idx = set()  # 存放有效结点
    # init que
    # goal: maximize the distance between clusters
    before_dis_clusters = 0.0
    for i in range(len(data)):
        dis_list = []
        valid_idx.add(i)
        for j in range(i + 1, len(data)):
            dis, iou = cal_dist(data[i], data[j])
            que.put((dis, (i, j, iou)))
            dis_list.append(dis)
            before_dis_clusters += dis

        dis_lists.append(np.array(dis_list))

    before_dis_clusters = before_dis_clusters

    new_cluster_idx = len(data)

    # 开始聚类
    while True:
        i, j, iou, d = None, None, None, None
        dis_list = []
        flag = False
        while not que.empty():
            item = que.get()
            d = item[0]
            i, j, iou = item[1]
            if i in valid_idx and j in valid_idx:
                flag = True
                break
        if i is None or not flag or iou == 0.0 or d > threshold: break  # 没有类可聚了
        # print(i, j, f'dis={d}')
        # i,j 聚类形成新类
        valid_idx.remove(i)
        valid_idx.remove(j)

        # 新簇的合成
        new_cluster = copy.deepcopy(data[i])
        new_cluster.add_cluster(data[j])

        dis_clusters = before_dis_clusters - sum(dis_lists[i]) - sum(dis_lists[j])
        for idx in valid_idx:
            dis, iou = cal_dist(new_cluster, data[idx])
            que.put((dis, (new_cluster_idx, idx, iou)))
            dis_clusters += dis
            dis_list.append(dis)

        # c = len(valid_idx) + 1
        # before_average_dis_clusters = before_dis_clusters / (c * (c + 1))
        # average_dis_clusters = dis_clusters / (c * (c - 1)) if c != 1 else sys
        #
        # if before_average_dis_clusters > average_dis_clusters:
        #     valid_idx.add(i)
        #     valid_idx.add(j)
        #     break

        dis_lists.append(np.array(dis_list))

        valid_idx.add(new_cluster_idx)
        new_cluster_idx += 1
        data.append(new_cluster)

    data = np.array(data)
    data = data[np.array(list(valid_idx))]

    return data


def cal_dist(cluster_1, cluster_2, eps=1e-8):
    x1_center, y1_center, z1, w1, h1, _ = cluster_1.get_message()
    x2_center, y2_center, z2, w2, h2, _ = cluster_2.get_message()

    dis = math.sqrt(
        math.pow(x1_center - x2_center, 2.0) + math.pow(y1_center - y2_center, 2.0) + math.pow(z1 - z2, 2.0))
    giou, iou = Giou((x1_center, y1_center, w1, h1), (x2_center, y2_center, w2, h2), True)
    result = dis / (giou + eps)
    return result, iou


if __name__ == '__main__':
    data = []
    with open('./data.txt', 'r') as f:
        for line in f.readlines():
            t = line.split(' ')
            c = Cluster(float(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4]), 0.5)
            data.append(c)
    print(f'len={len(data)}')
    data = HC(data)

    print('*'*50)
    for index, item in enumerate(data):
        print(f'{index+1}: ', end='')
        for i in item.get_message():
            print(i, end=', ')
        print(f'{sum(item.Conf)/len(item.Conf)}')

"""
Data:
E:/lung dataset/LUNA16/subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.259543921154154401875872845498.mhd
385 199 32 32 45
385 200 30 30 46
385 200 30 30 47
384 200 31 31 48
386 199 32 32 49
385 199 34 33 50
384 199 33 34 51
385 199 33 33 52
384 199 33 33 53
384 199 32 31 54
384 199 32 31 55
384 199 31 31 56
382 199 33 33 57
382 198 32 32 58
382 198 32 31 59
381 198 32 32 60
381 197 33 33 61
380 197 33 32 62
381 196 32 31 63
382 196 31 31 64
389 185 29 29 70
389 184 30 30 71
389 184 32 31 72
389 184 32 31 73
389 184 32 31 74
389 184 31 31 75
390 184 30 31 76
390 183 30 30 77
390 183 30 30 78
390 184 30 30 79
388 183 30 30 80
389 182 32 31 81
389 182 32 32 82
98 314 6 6 116
99 314 6 5 117
156 144 7 7 229
156 143 8 8 230
156 143 8 8 231
156 144 9 9 232
155 144 8 7 233
154 146 6 6 234
148 167 9 9 239
148 167 9 9 240
148 167 9 8 241
148 168 9 9 242
148 169 8 8 243
148 170 7 7 244
155 181 8 9 251
156 195 7 6 255
157 195 8 8 256
157 196 8 8 257
158 197 8 8 258
158 198 7 8 259
159 199 8 8 260
363 213 6 6 332
376 287 6 7 354
203 346 10 11 441
200 346 15 14 442
199 347 17 17 443
197 347 19 19 444
196 348 18 18 445
187 344 24 25 465
"""
