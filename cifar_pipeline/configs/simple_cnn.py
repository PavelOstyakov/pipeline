import random

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor

from pipeline.models.base import Flatten
from .base import ConfigCIFARBase

MODEL_SAVE_PATH = "models/cifar_simple_cnn"
BATCH_SIZE = 128

SEED = 85
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)


def get_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
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
    return model


class Config(ConfigCIFARBase):
    def __init__(self):
        model = get_model()
        transforms = ToTensor()
        super().__init__(model=model, model_save_path=MODEL_SAVE_PATH,
                         epoch_count=2, batch_size=BATCH_SIZE, transforms=transforms)
