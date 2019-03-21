from pipeline.core import PipelineError
from pipeline.utils import get_path

import torch.utils.data as data
from enum import auto

import os
import pandas as pd
import torch


class MNISTDataset(data.Dataset):
    MODE_TRAIN = auto()
    MODE_VAL = auto()

    def __init__(self, path, mode, val_ratio):
        path = get_path(path)
        if not os.path.exists(path):
            raise PipelineError("Path {} does not exist".format(path))

        dataset = pd.read_csv(path).values
        train_length = int(len(dataset) * (1 - val_ratio))
        if mode == self.MODE_TRAIN:
            dataset = dataset[:train_length]
        else:
            dataset = dataset[train_length:]

        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        row = self._dataset[item]

        image = row[1:].reshape(28, 28, 1).astype("uint8")
        target = torch.Tensor([float(row[0])]).float()
        return image, target


class MNISTImagesDataset(MNISTDataset):
    def __init__(self, path, mode, val_ratio):
        super().__init__(path, mode, val_ratio)

    def __getitem__(self, item):
        image, _ = super().__getitem__(item)
        return image


class MNISTTargetsDataset(MNISTDataset):
    def __init__(self, path, mode, val_ratio):
        super().__init__(path, mode, val_ratio)

    def __getitem__(self, item):
        _, target = super().__getitem__(item)
        return target
