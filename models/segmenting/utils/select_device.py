import torch


def select_device():
    cuda = torch.cuda.is_available()
    return torch.device('cuda:0' if cuda else 'cpu')
