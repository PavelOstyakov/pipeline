import torch.nn as nn
import torch


class ModelBase(nn.Module):
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


class FeatureExtractorModuleBase(ModelBase):
    def __init__(self, num_features):
        super().__init__()
        self._num_features = num_features

    def get_num_features(self):
        return self._num_features


class ClassificationModuleBase(ModelBase):
    def __init__(self, num_features):
        super().__init__()
        self._num_features = num_features

    def get_num_features(self):
        return self._num_features
