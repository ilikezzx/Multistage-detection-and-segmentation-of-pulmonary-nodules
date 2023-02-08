#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:周志勋

import sys
import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tkinter as tk
import tkinter.font
import tkinter.messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk

sys.path.append('..')
from progress_bar import pb
from plot3D import plot_3D
from loading import load_itk_image
from plot_panoramic import count_3D
from models import detect_position, detect_segmentation, detect_classify, detect_position_2, HC, Cluster
from model_selection import ModelSelection, ms_tkinter
from print_text import print_text

c3 = None
is_loading = False
raw_images = None
view_raw_images = []

selected_models = ModelSelection()


def load_images(l1, t1, window):
    global is_loading, raw_images, view_raw_images, c3
    mhd_path_ = askopenfilename(title='please select CT View')
    mhd_path = mhd_path_
    print(mhd_path)

    if mhd_path != "":
        if mhd_path[-4:] != '.mhd':
            tk.messagebox.showwarning(title='please select right suffix', message='请选择正确的CT文件，后缀格式：mhd')
            return
        p = pb(window, window_name='Loading CT Progress')
        is_loading = True
        raw_images, SP, OR, is_filp = load_itk_image(mhd_path)
        c3 = count_3D(ct_shape=raw_images.shape)

        for slice_index in range(raw_images.shape[0]):
            im = raw_images[slice_index] * 255.0
            im = Image.fromarray(im).resize((200, 200))
            # im = ImageTk.PhotoImage(im)
            view_raw_images.append(im)

            if slice_index % (round(raw_images.shape[0] * 1 / 20.0)) == 0:
                "控制改变数量，避免浪费的时间"
                p.update(slice_index / raw_images.shape[0], 1, stage_number=1)
        p.destroy()

        orient_img = ImageTk.PhotoImage(view_raw_images[0])
        l1.config(image=orient_img)
        l1.image = orient_img
        l1.update()
        # 需要连续改变两个引用
        print_text(t1, message='CT图像加载成功')
        window.update()
        # print("加载成功!!")


def view_images(l1, t1):
    global is_loading, view_raw_images
    if is_loading:
        for slice_index in range(len(view_raw_images)):
            # 减缓视频帧切换速度
            time.sleep(0.05)
            img = ImageTk.PhotoImage(view_raw_images[slice_index])

            l1.config(image=img)
            l1.image = img
            l1.update()
        print_text(t1, message='CT浏览结束！')
    else:
        # print("请先加载CT文件")
        print_text(t1, message='请先加载CT文件！')


def detecting_luna(f, t1, view_frame):
    global view_raw_images, raw_images
    if len(view_raw_images) == 0:
        print_text(t1, message='开始检测前,请先加载CT文件！')
        return

    if len(f.winfo_children()) > 1:
        "覆盖之前的CT检测数据"
        for i, widget in enumerate(f.winfo_children()):
            if i == 0:
                continue
            widget.destroy()  # 删除多余控件
    if len(view_frame.winfo_children()) > 1:
        for widget in view_frame.winfo_children()[1:]:
            widget.destroy()  # 删除多余控件

    p = pb(f, window_name='detecting CT Progress')
    # 2D 目标检测阶段
    valid_images = detect_position(imgs=raw_images, model_name=selected_models.detector, pb=p)

    # CNS算法
    centers = []
    for id, valid_image in enumerate(valid_images):
        for position in valid_image['positions']:
            center = [id, valid_image['index']]
            for item in position:
                center.append(item)
            center.append(valid_image['conf'].item())
            centers.append(center)

    clusters = []
    for center in centers:
        c = Cluster(center[2] * 512.0, center[3] * 512.0, center[4] * 512.0, center[5] * 512.0, center[1], center[6])
        clusters.append(c)
    clusters = HC(clusters, 20.0)
    new_centers = []
    for index, c in enumerate(clusters):
        item = c.get_message()
        new_centers.append([index, int(item[2]), round(item[0]) / 512, round(item[1]) / 512, round(item[3]) / 512,
                            round(item[4]) / 512])

    # 3D 目标二分类阶段
    "centers = [[id,z_index,x,y,w,h],...,]"
    true_patchs, centers, _ = detect_classify(images=raw_images, centers=new_centers, model_name=selected_models.classifier,
                                           pb=p)
    # 3D 图像分割阶段
    valid_images_2 = detect_segmentation(patchs=true_patchs, model_name=selected_models.divider, pb=p)

    p.destroy()
    # print(valid_images)
    number = len(centers)

    # 相册模板:也不知道为什么，这里设置height就会出现错误
    canvas = tk.Canvas(f, width=399, bg='#e3dedb')
    canvas.place(x=0, y=25, anchor='nw')
    f22 = tk.Frame(canvas, width=379, bg='#e3dedb')
    f22.place(x=0, y=0, anchor='nw')
    vbar = tk.Scrollbar(canvas, orient=tk.VERTICAL)
    vbar.place(x=379, width=20, height=275)
    vbar.configure(command=canvas.yview)

    for i in range(number):
        x = i % 2
        y = i // 2
        small_frame = tk.Frame(f22)
        img = ImageTk.PhotoImage(view_raw_images[centers[i][1]].resize((100, 100)))
        lx = tk.Button(small_frame, image=img, width=100, height=100)
        lx.config(command=lambda i=i: detail_detecting(f22, centers[i][1],
                                                       detecte_image=valid_images[centers[i][0]]['new_image'],
                                                       patch_message=valid_images_2[i]))
        lx.image = img
        lx.pack(side=tk.TOP)

        patch = valid_images_2[i][1]
        # center = centers[i][1:4]
        # center[0] = valid_images_2[i][2]
        center = valid_images_2[i][2]
        c3.update(center, patch)

        small_label = tk.Label(small_frame, text=f'Slice {centers[i][1] + 1}', font=('幼圆', 12))
        small_label.pack(side=tk.BOTTOM)
        small_frame.grid(row=y, column=x, pady=10, padx=25, sticky='new')

    f22.update()  # 更新副Frame
    canvas.config(yscrollcommand=vbar.set)  # 设置滚动条
    canvas.create_window((379 // 2, 0), window=f22)
    canvas.config(scrollregion=canvas.bbox(tk.ALL))
    canvas.update()

    # ct mask
    draw_images = c3.draw()
    plot_3D(view_frame, draw_images, shape=(299, 275))

    print_text(t1, message='检测完毕!')


def detail_detecting(window, slice_index, detecte_image, patch_message):
    """
    关于第二模块的弹窗
    :param patch_message: (patch, z_index)
    :param detecte_image: 2d 目标检测的结果
    :param window: 母窗体
    :param slice_index: 切片编号
    """
    global raw_images
    window_detail = tk.Toplevel(window)
    window_detail.geometry('550x550')
    window_detail.title(f'Slice {str(slice_index + 1)}')

    # 原图
    origin_image = raw_images[slice_index] * 255.0
    origin_image = Image.fromarray(origin_image).resize((200, 200))
    origin_image = ImageTk.PhotoImage(origin_image)
    # 定位图
    detecte_image = Image.fromarray(detecte_image.astype(np.uint8)).resize((200, 200))
    detecte_image = ImageTk.PhotoImage(detecte_image)
    # patch_message
    truth_patch, patch, _, z_index = patch_message
    # 不能直接乘，因为这里传的是引用，切块邻近的话用*=相互会影响。
    segment_image = Image.fromarray((patch * 255.0)[z_index]).resize((100, 100))
    segment_image = ImageTk.PhotoImage(segment_image)
    truth_image = Image.fromarray((truth_patch * 255.0)[z_index]).resize((100, 100))
    truth_image = ImageTk.PhotoImage(truth_image)

    l1 = tk.Label(window_detail, image=origin_image, bg='gray')
    l1.image = origin_image
    l1.place(x=50, y=50)
    ll1 = tk.Label(window_detail, text='原图', font=('幼圆', 15, tk.font.BOLD))
    ll1.place(x=125, y=260)

    l2 = tk.Label(window_detail, image=detecte_image)
    l2.image = detecte_image
    l2.place(x=300, y=50)
    ll2 = tk.Label(window_detail, text='定位图', font=('幼圆', 15, tk.font.BOLD))
    ll2.place(x=370, y=260)

    l3 = tk.Label(window_detail, image=segment_image)
    l3.image = segment_image
    l3.place(x=150, y=350)

    l4 = tk.Label(window_detail, image=truth_image)
    l4.image = truth_image
    l4.place(x=50, y=350)
    ll3 = tk.Label(window_detail, text='切片原图/语义分割图', font=('幼圆', 15, tk.font.BOLD))
    ll3.place(x=50, y=510)

    f = tk.Frame(window_detail)
    plot_3D(f, patch * 255.0, isPack=True)
    f.place(x=300, y=300)
    f.update()
    ll4 = tk.Label(window_detail, text='切块图像分割图', font=('幼圆', 15, tk.font.BOLD))
    ll4.place(x=330, y=510)


def main():
    """创建窗体"""
    window = tk.Tk()  # 母窗体
    window.title("Multi-stage detection and segmentation of pulmonary nodules")
    window.geometry('1000x450')
    """第一模块"""
    f1 = tk.Frame(window, height=300, width=300, bg='#e3dedb')
    f1.place(x=0, y=0, anchor='nw')
    none_image_1 = Image.open('../image/none-select.png').resize((200, 200))
    none_image_1 = ImageTk.PhotoImage(none_image_1)
    l1 = tk.Label(f1, image=none_image_1)
    l1.place(x=50, y=5, anchor='nw')

    b1 = tk.Button(f1, text='Load CT', font=('Arial', 12), width=10, height=1,
                   command=lambda: load_images(l1, t1, window))
    b1.place(x=45, y=220, anchor='nw')

    b2 = tk.Button(f1, text='View CT', font=('Arial', 12), width=10, height=1, command=lambda: view_images(l1, t1))
    b2.place(x=160, y=220, anchor='nw')

    b3 = tk.Button(f1, text='Start Detecting', font=('Arial', 12), width=23, height=1,
                   command=lambda: detecting_luna(f2, t1, f3))
    b3.place(x=45, y=260, anchor='nw')

    """第二模块"""
    f2 = tk.Frame(window, height=300, width=399, bg='#e3dedb')
    f2.place(x=301, y=0, anchor='nw')
    l2 = tk.Label(f2, text="2D-目标检测可疑肺结节结果", font=('幼圆', 15, tk.font.BOLD))
    l2.place(x=75, y=0, anchor='nw')

    """第三模块"""
    f3 = tk.Frame(window, height=300, width=299, bg='#e3dedb')
    f3.place(x=701, y=0, anchor='nw')
    l3 = tk.Label(f3, text="3D-CT切块图像分割结果", font=('幼圆', 15, tk.font.BOLD))
    l3.place(x=45, y=0, anchor='nw')

    """显示屏模块"""
    f4 = tk.Frame(window, height=150, width=1000, bg='white')
    f4.place(x=0, y=300, anchor='nw')
    t1 = tk.Text(f4, height=150, width=1000, font=('Arial', 12), bg='white')
    scroll = tk.Scrollbar(t1)
    scroll.config(command=t1.yview)
    scroll.place(x=980, y=0, width=20, height=280)
    t1.place(x=0, y=0, anchor='nw')
    t1.config(state='disabled', yscrollcommand=scroll.set)

    """菜单栏"""
    menubar = tk.Menu(window)
    filemenu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='帮助', menu=filemenu)
    filemenu.add_command(label='模型选择', command=lambda: ms_tkinter(window, selected_models))
    filemenu.add_command(label='关于我们')
    menubar.add_command(label='退出', command=window.quit)
    window.config(menu=menubar)  # 加上这代码，才能将菜单栏显示

    window.resizable(0, 0)  # 防止用户调整尺寸
    window.mainloop()


if __name__ == '__main__':
    main()
