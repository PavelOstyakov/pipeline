from .base import FeatureExtractorModuleBase


class Resnet18Model(FeatureExtractorModuleBase):
    NUM_FEATURES = 512

    def __init__(self):
        super().__init__(num_features=Resnet18Model.NUM_FEATURES)
        raise NotImplementedError
