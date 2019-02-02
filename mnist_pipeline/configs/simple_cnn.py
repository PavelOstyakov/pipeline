from .base import ConfigMNISTBase

from pipeline.models.base import Flatten

import torch.nn as nn


class Config(ConfigMNISTBase):
    def __init__(self, model_save_path="models/simple_cnn"):
        model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(128, 10)
        )

        super().__init__(model=model, model_save_path=model_save_path)
