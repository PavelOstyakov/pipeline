from .base import ConfigImageNetBase

from torch.nn import DataParallel

from torchvision.models import resnet50

MODEL_SAVE_PATH = "models/imagenet_resnet_50"


class Config(ConfigImageNetBase):
    def __init__(self, model_save_path=MODEL_SAVE_PATH):
        super().__init__(model=DataParallel(resnet50()), model_save_path=model_save_path)
