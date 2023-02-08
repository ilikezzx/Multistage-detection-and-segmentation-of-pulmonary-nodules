"""
    分类器
"""

import time
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from ..config import classification_information, classification_transform, classification_batch_size
from .dataset import ClassifierDataset
from .models import VGG, ResXNet, ResNet_basic111_maxpool, ResNet_small_basic111_maxpool, \
    ResXNet_2
from .crop import crop_patch


def get_model(model_name):
    if model_name == "ResXNet":
        model = ResXNet
    elif model_name == "ResNet_basic111_maxpool":
        model = ResNet_basic111_maxpool
    else:
        model = None

    assert model is not None, "Please check classifier's model_name"
    return model


def detect_classify(images, centers, model_name='ResNet_basic111_maxpool', pb=None):
    patchs = np.array([crop_patch(images, center, 64, 64) for center in centers])
    if model_name is None:
        model_name = 'In-ResNet'

    # model_name = "VGG"

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    model_weight = classification_information[model_name]['weight']
    Model = get_model(model_name)

    # 构建模型
    Arch64 = Model(1, [(1, 32), (1, 64), (1, 64), (1, 64)])
    Arch24 = ResNet_small_basic111_maxpool(1, [(1, 32), (1, 64), (1, 64)])

    if model_weight is not None:
        # print('loading weights...')
        Arch64.load(model_weight)
        Arch24.load(r'../models/classifing/weights/classifier/cp_resnet_24@3/best.pth')

    Arch64.to(device)
    Arch64.eval()

    Arch24.to(device)
    Arch24.eval()

    dataset = ClassifierDataset(patchs=patchs[:, 0], transform=classification_transform)
    dataloader = DataLoader(dataset, batch_size=classification_batch_size, shuffle=False, num_workers=1)

    t0 = time.time()
    answers = np.array([])
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            patch, _ = data
            patch = patch.to(device)

            small_patch = patch[:, :, 20:44, 20:44, 20:44]
            pred = Arch64(patch)  # [batch_size, 2]
            # pred_3 = model_3(small_patch)
            pred_2 = Arch24(small_patch)
            # pred_4 = model_4(patch)
            # _, predicted = torch.max(pred.data, 1)  # [batch_size]
            # predicted = predicted.detach().cpu().numpy()

            # print(pred)   # modify
            # prob = np.array([((p[1]) / (p[0] + p[1])).detach().cpu().numpy() for p in pred]) # 二分类
            # prob = np.array([0.25 * p1[0].detach().cpu().numpy() + 0.25 * p2[0].detach().cpu().numpy() + 0.25 * p3[
            #     0].detach().cpu().numpy() + 0.25 * p4[0].detach().cpu().numpy() for p1, p2, p3, p4 in
            #                  zip(pred, pred_2, pred_3, pred_4)])  # 单分类
            # prob = np.array([p[0].detach().cpu().numpy() for p in pred])
            prob = np.array([0.7*p1[0].detach().cpu().numpy()+0.3*p2[0].detach().cpu().numpy() for p1, p2 in zip(pred, pred_2)])
            # print(prob)  # modify
            answers = np.concatenate((answers, prob), axis=0)  # modify

            if pb is not None and index % max(round(len(dataloader) * 3 / 20.0), 1) == 0:
                pb.update(index / len(dataloader), 2, 3)

    assert answers.shape[0] == len(patchs), "please check  your  classifier"
    true_patchs = []
    true_centers = []
    probs = []  # modify

    for index, item in enumerate(answers):
        if item >= 0.5:  # modify
            true_patchs.append(patchs[index])
            # print(patchs[index, 0].shape)
            # image1 = patchs[index, 0][31]
            # image2 = patchs[index, 0][32]
            # image3 = patchs[index, 0][33]
            #
            # image_small_1 = image1[20:44, 20:44]
            # image_small_2 = image2[20:44, 20:44]
            # image_small_3 = image3[20:44, 20:44]
            #
            # imshow(np.hstack([image1, image2, image3]))
            # imshow(np.hstack([image_small_1, image_small_2, image_small_3]))

            true_centers.append(centers[index])
            probs.append(item)  # modify
    t1 = time.time()
    print(f'classification time:{t1 - t0}, patch_num:{len(centers)}, every patch average={(t1 - t0) / len(centers)}')

    return true_patchs, true_centers, probs  # modify


def imshow(image):
    import cv2
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
