"""
    这一文件关于绘制 3D切块--图像分割后的结果
"""

import os
import cv2
import numpy as np
import tkinter as tk
import matplotlib
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def draw(img, threshold=-400):
    # matplotlib.use('TkAgg')  # 使用组件
    if np.max(img) == 0.0:
        img[0, 0, 0] = 1.0
    verts, faces, x, y = measure.marching_cubes(img, threshold)

    fig = plt.figure(figsize=(10, 10))  # 设置窗体大小
    ax = fig.add_subplot(111, projection='3d', xticks=[], yticks=[])

    mesh = Poly3DCollection(verts[faces], alpha=0.8)
    face_color = [0.8, 0, 0]  # 使用的颜色 rgb(0~1)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.patch.set_facecolor("gray")
    # 设置坐标轴
    ax.set_xlim(0, img.shape[0])
    ax.set_ylim(0, img.shape[1])
    ax.set_zlim(0, img.shape[2])
    # plt.axis('off')
    # plt.show()
    return fig


def plot_3D(frame, patch, shape=(200, 200), isPack=False):
    matplotlib.use('TkAgg')  # 使用组件
    fig = draw(patch, 0)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().config(width=shape[0], height=shape[1])
    if not isPack:
        canvas.get_tk_widget().place(x=0, y=25)
    else:
        canvas.get_tk_widget().pack(side=tk.BOTTOM)
    frame.update()
