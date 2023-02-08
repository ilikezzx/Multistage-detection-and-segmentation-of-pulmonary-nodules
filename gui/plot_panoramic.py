"""
    这一个文件写的是 3D全景 CT前期的统计
"""

import numpy as np


class count_3D(object):
    def __init__(self, ct_shape=(64, 64, 64)):
        super(count_3D).__init__()

        temp = {'tot': 0, 'correct': 0}
        self.ct_shape = ct_shape
        self.stat_tot = np.zeros(ct_shape)
        self.stat_tot += 1e-9
        self.stat_correct = np.zeros(ct_shape)

    def update(self, center, patch):
        """
        :param center: the center of the CT
        :param patch: the patch of Value is 0 or 1
        :return:
        """
        z_center, x_center, y_center = center
        depth_patch, height_patch, width_patch = patch.shape
        # x_center = round(x_center * self.ct_shape[2])
        # y_center = round(y_center * self.ct_shape[1])

        # print(patch.shape,center)
        # print(x_center, width_patch,x_center - width_patch // 2, x_center + width_patch // 2)

        self.stat_correct[z_center - depth_patch // 2:z_center + depth_patch // 2,
        y_center - height_patch // 2: y_center + height_patch // 2,
        x_center - width_patch // 2: x_center + width_patch // 2] += patch

        # self.stat_tot[z_center - depth_patch // 2:z_center + depth_patch // 2,
        # y_center - height_patch // 2: y_center + height_patch // 2,
        # x_center - width_patch // 2: x_center + width_patch // 2] += 1

    def draw(self):
        """
            只有超过 1/2的CT分割图 显示是肺结节像素点，该结点像素点才显示出来
        """
        segmentation_images = np.zeros(self.ct_shape)
        # segmentation_images[(self.stat_correct / self.stat_tot) >= 0.5] = 255.0
        segmentation_images[self.stat_correct >= 1] = 255.0
        return segmentation_images
