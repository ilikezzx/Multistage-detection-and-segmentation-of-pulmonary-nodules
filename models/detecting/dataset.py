import random
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
from .utils.torch_utils import select_device

device = select_device('')
half = device.type != 'cpu'  # half precision only supported on CUDA


class DetectorDataset(data.Dataset):
    def __init__(self, slices, transform=None):
        super(DetectorDataset, self).__init__()
        self.images = []
        for slice in slices:
            img = Image.fromarray(slice).resize((640, 640))
            img = np.array(img, dtype=np.float)
            img = np.stack((img,) * 3, axis=-1)

            self.images.append(img)

        self.transform = transform

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        img = torch.from_numpy(self.images[index]).to(device, non_blocking=True).float()
        img = img.half() if half else img.float()
        img = img.permute(2, 0, 1)

        label = int(1)
        return img, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.images)
