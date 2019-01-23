import torch.utils.data as data


class EmptyDataset(data.Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, item):
        assert False, "This code is unreachable"
