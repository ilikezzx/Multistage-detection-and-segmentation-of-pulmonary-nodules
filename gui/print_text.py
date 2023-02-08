import time


def print_text(t1, message):
    t1.config(state='normal')
    t1.insert('insert', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + f"   ---  {message}\n")
    t1.update()
    t1.config(state='disabled')
