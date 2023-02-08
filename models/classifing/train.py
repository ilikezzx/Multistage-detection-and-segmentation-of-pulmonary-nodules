import os
import time
import yaml
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from models import Classifier_1,Classifier_2
from dataset import ClassifierDataset
from torchvision import transforms

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

best_acc = 89.0

def recording_results(file_path, epoch_num, train_avg_loss, test_avg_acc):
    with open(file_path, 'a+') as f:
        results = {
            epoch_num: {
                'train_avg_loss': train_avg_loss,
                'test_avg_acc': test_avg_acc
            }
        }
        yaml.dump(results, f)

def train(weight=None, epochs=50+33, test_frequency=1):
    transform = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.ToTensor()
        ])
    }

    model = Classifier_2(in_channel=1, out_channels=[32, 64, 64, 64])
    if weight is not None:
        model.load(weight)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6)

    trainset = ClassifierDataset(r'/content/train_local/classifier/train.txt', transform=transform['train'])
    testset = ClassifierDataset(r'/content/train_local/classifier/test.txt', transform=transform['test'])

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)

    for e in range(33,epochs):
        print(f'--第{e + 1}轮开始--')
        running_loss = 0.0
        model.train()
        for data in tqdm(trainloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print(loss.item())
        print(f'Total_loss={running_loss},average loss={running_loss / len(trainloader)*4}')
        # if e % test_frequency:
        acc=test(testloader, model)
        recording_results('./classifier_2.yaml', e + 1, running_loss / len(trainloader), acc)


def test(testloader, model):
    global best_acc
    model.eval()
    correct = 0
    total = 0
    print("--开始测试--")
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            # print(predicted)
            # print(labels)
        acc = (100 * correct / total).item()
        print('Accuracy on the test set:%d %%' % acc)
        if best_acc < acc:
            best_acc = acc
            model.save(name=f"./best_{str(round(acc))}_{model.model_name}.pth")
        model.save(name=f"./last_{model.model_name}.pth")
    return acc


if __name__ == '__main__':
    train(weight=r'/content/last_Classifier_2.pth')
