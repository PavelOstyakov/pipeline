from .base import ConfigImageNetBase

from torch.nn import DataParallel


from pipeline.models.image_models.resnet_fixup import resnet50

MODEL_SAVE_PATH = "models/imagenet_resnet_50_fixup"


class Config(ConfigImageNetBase):
    def __init__(self, model_save_path=MODEL_SAVE_PATH):
        super().__init__(model=DataParallel(resnet50()), model_save_path=model_save_path, use_mixup=True, batch_size=128 * 7, learning_rate=0.1 * 7)
