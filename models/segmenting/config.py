"""
    存放 3D图像分割的超参数
"""

import argparse

information = {'Base': r"../models/segmenting/weights/base.pt",
               'Res': r"../models/segmenting/weights/res.pt"}

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Base')
    parser.add_argument('--weights', nargs='+', type=str, default=r'../models/segmenting/weights/base.pt',
                        help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--out-channels', type=int, default=1)
    parser.add_argument('--f-maps', type=int, default=32)

    opt = parser.parse_args()
    # print(opt)
    return opt
