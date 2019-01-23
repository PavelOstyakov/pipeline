from .base import ClassificationModuleBase

import torch.nn as nn


class ClassificationModuleLinear(ClassificationModuleBase):
    def __init__(self, num_features, num_classes):
        super().__init__(num_features)

        self.out = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.out(x)
