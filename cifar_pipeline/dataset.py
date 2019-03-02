import torch.utils.data as data
from torchvision.datasets.cifar import CIFAR10


class CIFARDataset(data.Dataset):
    def __init__(self, path, download=True, train=True):
        self._dataset = CIFAR10(path, download=download, train=train)

    def get_image(self, item):
        return self._dataset[item][0]

    def get_class(self, item):
        return self._dataset[item][1]

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        return self._dataset[item]


class CIFARImagesDataset(CIFARDataset):
    def __getitem__(self, item):
        return self.get_image(item)


class CIFARTargetsDataset(CIFARDataset):
    def __getitem__(self, item):
        return self.get_class(item)
