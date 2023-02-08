"""
    进度条组件
    写一个window，利用TopLevel创建额外窗体，用来显示加载和检测进度
"""

import time
import tkinter as tk


class pb(object):
    def __init__(self, mother_window, window_name='实时进度', window_geometry='650x100', **kw):
        super().__init__()
        self.frame = tk.Toplevel(mother_window)
        self.frame.title(window_name)
        self.frame.geometry(window_geometry)

        self.l1 = tk.Label(self.frame, text=f'Progress:')
        self.l1.place(x=40, y=40)

        self.l2 = tk.Label(self.frame, text='0%')
        self.l2.place(x=580, y=40)

        self.canvas = tk.Canvas(self.frame, width=465, height=22, bg='white')
        self.canvas.place(x=110, y=40)
        self.fill_line = self.canvas.create_rectangle(1.5, 1.5, 0, 23, width=0, fill='green')
        # x代表有多少格子,n代表每格的宽度
        self.x = 100
        self.n = 465 / self.x

    def update(self, this_stage, now_stage, stage_number):
        """
        :param this_stage: 这一阶段的进度
        :param stage_number: 有几个阶段
        :return:
        """
        assert stage_number >= 1, "stage number MUST bigger than 1"
        assert 1 >= this_stage >= 0, "this_stage MUST smaller than 1 bigger than 0"
        this_stage /= stage_number
        this_stage += (now_stage-1)/stage_number
        now_pb = this_stage * self.x * self.n
        self.canvas.coords(self.fill_line, (0, 0, now_pb, 60))
        self.l2.config(text=f'{round(this_stage * 100.0)}%')
        self.frame.update()
        time.sleep(0.02)

    def destroy(self):
        self.frame.destroy()


if __name__ == '__main__':
    window = tk.Tk()
    window.title('111')
    window.geometry('600x200')
    p = pb(window)

    time.sleep(3)
    for i in range(100):
        p.update(i / 100, 1,1)
