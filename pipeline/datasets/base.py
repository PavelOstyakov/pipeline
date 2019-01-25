import torch.utils.data as data
import torch


class EmptyDataset(data.Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, item):
        assert False, "This code is unreachable"


class DatasetComposer(data.Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        self._dataset_length = len(datasets[0])
        for dataset in datasets:
            assert self._dataset_length == len(dataset)

    def __len__(self):
        return self._dataset_length

    def __getitem__(self, item):
        return tuple(dataset[item] for dataset in self._datasets)


class OneHotTargetsDataset(data.Dataset):
    def __init__(self, targets, class_count):
        self._targets = targets
        self._class_count = class_count

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, item):
        target = self._targets[item]
        result = torch.zeros(self._class_count, dtype=torch.float32)
        result[target] = 1
        return result


class MultiLabelTargetsDataset(data.Dataset):
    def __init__(self, targets, class_count):
        self._targets = targets
        self._class_count = class_count

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, item):
        target = self._targets[item]
        result = torch.zeros(self._class_count, dtype=torch.float32)

        for class_id in target:
            result[class_id] = 1

        return result
