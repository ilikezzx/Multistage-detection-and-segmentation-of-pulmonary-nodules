"""模型选择，封装成类"""


class ModelSelection(object):
    def __init__(self):
        # 定位器
        self.detector = 'Yolo'
        # 分类器
        self.classifier = 'ResNet_basic111_maxpool'
        # 分割器
        self.divider = 'Res'

    def change_model(self, detector, classifier, divider):
        if detector is not None:
            self.detector = detector
        if classifier is not None:
            self.classifier = classifier
        if divider is not None:
            self.divider = divider


"""模型选择 窗体"""
import tkinter as tk
from tkinter import ttk


def change_model(ms, detector, classifier, divider):
    ms.change_model(detector, classifier, divider)


def ms_tkinter(window, ms):
    ms_tk = tk.Toplevel(window)
    ms_tk.title("model selection")
    ms_tk.geometry('325x150')

    """定位器选择"""
    l1 = tk.Label(ms_tk, text="①定位器",font=('幼圆', 15, tk.font.BOLD))
    l1.place(x=20, y=20)
    detector = tk.StringVar()
    detector_r1 = tk.Radiobutton(ms_tk, text='Yolo', variable=detector, value='Yolo', command=lambda: change_model(ms,
                                                                                                                   detector.get(),
                                                                                                                   classifier.get(),
                                                                                                                   divider.get()))
    detector_r1.place(x=25, y=50)
    detector_r1.select()

    """分类器"""
    l2 = tk.Label(ms_tk, text="②分类器", font=('幼圆', 15, tk.font.BOLD))
    l2.place(x=120, y=20)
    classifier = tk.StringVar()
    classifier_r1 = tk.Radiobutton(ms_tk, text='ResNet', variable=classifier, value='ResNet',
                                   command=lambda: change_model(ms,
                                                                detector.get(),
                                                                classifier.get(),
                                                                divider.get()))
    classifier_r1.place(x=125, y=50)
    classifier_r1.select()

    classifier_r1 = tk.Radiobutton(ms_tk, text='In-ResNet', variable=classifier, value='In-ResNet',
                                   command=lambda: change_model(ms,
                                                                detector.get(),
                                                                classifier.get(),
                                                                divider.get()))
    classifier_r1.place(x=125, y=100)
    """分割器"""
    l3 = tk.Label(ms_tk, text="③分割器", font=('幼圆', 15, tk.font.BOLD))
    l3.place(x=220, y=20)
    divider = tk.StringVar()
    divider_r1 = tk.Radiobutton(ms_tk, text='Base', variable=divider, value='Base', command=lambda: change_model(ms,
                                                                                                                     detector.get(),
                                                                                                                     classifier.get(),
                                                                                                                     divider.get()))
    divider_r1.place(x=225, y=50)
    divider_r2 = tk.Radiobutton(ms_tk, text='Res', variable=divider, value='Res',
                                command=lambda: change_model(ms,
                                                             detector.get(),
                                                             classifier.get(),
                                                             divider.get()))
    divider_r2.place(x=225, y=100)
    divider_r1.select()
