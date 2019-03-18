import torch.utils.data as data
import random
import numpy as np


class MixUpDatasetWrapper(data.Dataset):
    def __init__(self, dataset, alpha=1):
        super().__init__()
        self._dataset = dataset
        self._alpha = alpha

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        first = self._dataset[item]
        second = random.choice(self._dataset)

        coeff = np.random.beta(self._alpha, self._alpha)

        result = []
        for elem1, elem2 in zip(first, second):
            result.append(elem1 * coeff + elem2 * (1 - coeff))

        return tuple(result)
