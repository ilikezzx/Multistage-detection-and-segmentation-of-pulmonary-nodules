"""
    常用函数库
"""
import SimpleITK as sitk
import numpy as np


def load_itk_image(mhd_path):
    image, SP, OR, is_filp = load_image(mhd_path)
    image = truncate_num(image)
    image = normalazation(image)

    return image, SP, OR, is_filp


def load_image(image_path):
    """
        合法加载tik文件
    :param image_path: raw文件路径
    :return: numpy矩阵
    """
    with open(image_path) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]

        transform = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transform = np.round(transform)

        # 判断是否是正向摆放
        if np.any(transform != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isfilp = True
        else:
            isfilp = False

    itk_image = sitk.ReadImage(image_path)
    SP = itk_image.GetSpacing()
    OR = itk_image.GetOrigin()
    np_itk_image = sitk.GetArrayFromImage(itk_image)

    if isfilp:
        np_itk_image = np_itk_image[:, ::-1, ::-1]  # 翻转

    return np_itk_image, SP, OR, isfilp


def truncate_num(image_array):
    """
        此函数是为了将CT值过大和过小的数据置为0
    """
    image_array[image_array > 400] = 400
    image_array[image_array < -1000] = -1000
    return image_array


def normalazation(image_array):
    """
        此函数的作用是归一化CT图像矩阵，将值范围缩小为[0,1]之间
    """
    image_array = image_array.astype(np.int)
    maxValue = image_array.max()
    minValue = image_array.min()

    # 归一化
    image_array = (image_array - int(minValue)) / (int(maxValue) - int(minValue))
    # avg = np.mean(image_array)
    # std = np.std(image_array)
    # image_array = (image_array - avg) / std

    return image_array
