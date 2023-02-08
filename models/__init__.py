"""
    这个文件夹主要为了存放3类模型文件
    ①2Ddetection：2D 关于肺结节切片目标检测模型文件夹，存放 Yolov5s/m 、 FastRCNN 模型
    ②3Dclassifying：3D 关于肺结节切块目标目标分类模型文件夹，存放
    ③3Dsegmenting：3D 关于正阳性肺结节切块目标图像分割模型文件夹，存放UNet、ResUNet、CLUNet 模型
"""
from .detecting import detect_position, detect_position_2
from .segmenting import detect_segmentation
from .classifing import detect_classify
from .CNS import HC, Cluster

__all__ = ['detect_position', 'detect_segmentation', 'detect_classify', 'detect_position_2', 'HC', 'Cluster']
