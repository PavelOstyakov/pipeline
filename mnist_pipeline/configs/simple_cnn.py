from .base import ConfigMNISTBase, PredictConfigMNISTBase

from pipeline.models.base import Flatten

import torch.nn as nn


MODEL_SAVE_PATH = "models/simple_cnn"


def get_model():
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
        nn.Linear(128, 1)
    )
    return model


class Config(ConfigMNISTBase):
    def __init__(self, model_save_path=MODEL_SAVE_PATH):
        super().__init__(model=get_model(), model_save_path=model_save_path)


class PredictConfig(PredictConfigMNISTBase):
    def __init__(self, model_save_path=MODEL_SAVE_PATH):
        super().__init__(model=get_model(), model_save_path=model_save_path)
