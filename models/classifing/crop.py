"""
    crop_patch：
        -- input：切片中心
        -- output：切块 [64:64:64] ~ [depth,height,width]
"""
import numpy as np
import random


def crop_patch(images, center, depth=64, radius=64):
    """
    :param radius:
    :param depth:
    :param images:  CT文件 [depth,height,width]
    :param center: 中心位置 [z_index,x,y,w,h],其中后四个是归一化后的结果
    :return:
        patch  [depth,height,width]
        the after randoming of the index of z_center
    """
    _, z_center, x_center, y_center, w, h = center
    depth_img, h_img, w_img = images.shape[:3]
    x_center, w = round(x_center * w_img), round(w * w_img)
    y_center, h = round(y_center * h_img), round(h * h_img)

    # 肺结节标注框的长宽
    max_r = min(w, h) // 2
    # 肺结节标注框的高
    max_d = min(min(z_center, depth_img - z_center), max_r)
    # print(z_center, x_center, y_center, w, h)
    if z_center + depth // 2 > depth_img:
        _z_center = depth_img - depth // 2
    elif z_center < depth // 2:
        _z_center = depth // 2
    else:
        _z_center = z_center

    if y_center + radius // 2 > h_img:
        _y_center = h_img - radius // 2
    elif y_center < radius // 2:
        _y_center = radius // 2
    else:
        _y_center = y_center

    if x_center + radius // 2 > w_img:
        _x_center = w_img - radius // 2
    elif x_center < radius // 2:
        _x_center = radius // 2
    else:
        _x_center = x_center

    real_center = [int(_z_center), _y_center, _x_center]
    # print(real_center)

    patch = images[real_center[0] - depth // 2:real_center[0] + depth // 2,
            real_center[1] - radius // 2:real_center[1] + radius // 2,
            real_center[2] - radius // 2:real_center[2] + radius // 2]

    assert patch.shape == (depth, radius, radius), "Wrone Shape,Please Check it crop.py in 42Line"
    return np.array([patch, real_center, z_center - (_z_center - depth // 2)], dtype=object)
