"""
    存放 2D-Detecting的超参数
"""
import argparse

information = {
    'Yolo': {
        'weights': r'../models/detecting/weights/yolo_2d.pt',
        'cfg': r'../models/detecting/data/model.yaml'
    }
}


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'./models/detecting/weights/yolov5l.pt',
                        help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='./models/detecting/data/yolov5l.yaml',
                        help='hyperparameters path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--batch-size', type=int, default=16, help='batch value')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    # print(opt)
    return opt
