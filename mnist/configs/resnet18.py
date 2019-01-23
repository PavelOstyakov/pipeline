from .base import ConfigMNISTBase

from pipeline.models.classification import ClassificationModuleLinear
from pipeline.models.image_classification import Resnet18Model

import torch.nn as nn


class Config(ConfigMNISTBase):
    def __init__(self):
        model = nn.Sequential(
            Resnet18Model(),
            ClassificationModuleLinear(Resnet18Model.NUM_FEATURES, 10)
        )

        super().__init__(model=model)
