from .base import ConfigMNISTBase

from pipeline.models.image_models.encoders.resnet import Resnet18FeatureExtractor

import torch.nn as nn


class Config(ConfigMNISTBase):
    def __init__(self, model_save_path="models/resnet18"):
        model = nn.Sequential(
            Resnet18FeatureExtractor(input_channels=1),
            nn.Linear(Resnet18FeatureExtractor.NUM_FEATURES, 10)
        )

        super().__init__(model=model, model_save_path=model_save_path)
