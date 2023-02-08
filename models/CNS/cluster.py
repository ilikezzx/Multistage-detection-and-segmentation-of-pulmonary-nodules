#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DBCSCAN 
@Author  ：zzx
@Date    ：2022/1/12 13:21 
"""


def list_mul(l1, l2):
    func = lambda x, y: x * y
    return sum(list(map(func, l1, l2)))


class Cluster(object):
    def __init__(self, x_center, y_center, w, h, z, conf):
        self.z_center = z
        self.x_center = x_center  # [x, y, w, h] range_value [0~512]
        self.y_center = y_center
        self.w = w
        self.h = h
        self.conf = conf

        self.Conf = [conf]
        self.X = [x_center]
        self.Y = [y_center]
        self.W = [w]
        self.H = [h]
        self.Z = [z]

    def get_message(self):
        return self.x_center, self.y_center, self.z_center, self.w, self.h, self.conf

    def add_cluster(self, cluster):
        self.Conf = self.Conf + cluster.Conf
        self.X = self.X + cluster.X
        self.Y = self.Y + cluster.Y
        self.W = self.W + cluster.W
        self.H = self.H + cluster.H
        self.Z = self.Z + cluster.Z

        # update centerid
        self.z_center = list_mul(self.Conf, self.Z) / sum(self.Conf)
        self.x_center = list_mul(self.Conf, self.X) / sum(self.Conf)
        self.y_center = list_mul(self.Conf, self.Y) / sum(self.Conf)
        self.w = list_mul(self.Conf, self.W) / sum(self.Conf)
        self.h = list_mul(self.Conf, self.H) / sum(self.Conf)
        self.conf = sum(self.Conf) / len(self.Conf)

if __name__ == '__main__':
    import numpy as np
    p1 = np.array([1, 2, 3])
    p2 = np.array([0,1])

    print(p1[p2])


