import random
import numpy as np
import torch.utils.data as data


class ClassifierDataset(data.Dataset):
    def __init__(self, patchs, transform=None):
        super(ClassifierDataset, self).__init__()
        self.images = np.array(patchs)
        self.transform = transform

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        patch = np.array(self.images[index])
        patch = patch.astype(np.float32)
        label = int(1)

        patch = self.transform(patch)
        while len(patch.shape) < 4:
            patch = patch.unsqueeze(0)
        # label = self.transform(label)

        return patch, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.images)
