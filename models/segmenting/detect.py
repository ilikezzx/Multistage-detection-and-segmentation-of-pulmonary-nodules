"""
    图像分割
"""

import cv2
import time
import torch
import numpy as np

from .utils import crop_patch, select_device
from .models import UNet3D, ResidualUNet3D, load_checkpoint

from ..config import load_segmentation_config, segmentation_information


def get_model(model_name, opt):
    assert model_name in ['Base', 'Res'], 'please input correct model name'
    model = None
    if model_name == 'Base':
        model = UNet3D(in_channels=opt.in_channels, out_channels=opt.out_channels, num_groups=1, f_maps=opt.f_maps, testing=True)
    elif model_name == 'Res':
        model = ResidualUNet3D(in_channels=opt.in_channels, out_channels=opt.out_channels, num_groups=8,
                               f_maps=opt.f_maps, layer_order='gcr',
                               testing=True)

    return model


def detect_segmentation(patchs, model_name=None, pb=None):
    opt = load_segmentation_config()
    device = select_device()
    # mask = np.zeros(images.shape).astype(np.uint8)
    print(model_name)
    model = get_model(model_name, opt)
    model = model.to(device)
    model.eval()
    load_checkpoint(segmentation_information[model_name], model)

    t0 = time.time()
    segmentPatchs = []
    for index, (patch, real_center, c) in enumerate(patchs):
        origin_patch = patch.copy()
        patch = torch.from_numpy(patch).float()
        patch = patch.to(device)

        z_index, y_index, x_index = real_center
        # patch[depth,height,width] ==> [batch_size,in_channels,depth,height,width]
        while len(patch.shape) < 5:
            patch = patch.unsqueeze(0)

        assert patch.shape == (1, 1, 64, 64, 64), 'Wrone shape,please check'

        pred = model(patch)
        pred = pred.view(-1, 64, 64)
        pred = pred.detach().cpu().numpy()

        # 二值化
        pred[pred < 0.5] = 0.0
        pred[pred >= 0.5] = 1.0
        pred = pred.astype(np.long)

        # 掩膜相乘，继续二值化
        pred = origin_patch * pred
        pred[pred >= 0.4] = 1.0
        pred[pred < 0.4] = 0.0

        segmentPatchs.append((origin_patch, pred, real_center, c))

        if index % max(round(len(patchs) * 3 / 20.0), 1) == 0 and pb is not None:
            pb.update(index / len(patchs), 3, 3)

        # for d in range(origin_patch.shape[0]):
        #     img = np.hstack([origin_patch[d], pred[d]])
        #     cv2.imshow('imgg', img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # break

    t1 = time.time()
    if len(patchs) > 0:
        print(f'segmentation time:{t1 - t0},patch_num:{len(patchs)}, every patch average={(t1 - t0) / len(patchs)}')
    else:
        print("sorry, no patches")
    return segmentPatchs


if __name__ == '__main__':
    images = np.load('1.npy')
    detect_segmentation(images,
                        [[32, 32 / images.shape[1], 32 / images.shape[2], 10 / images.shape[1], 10 / images.shape[2]]])
