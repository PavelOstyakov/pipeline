from .base import ConfigCIFARBase

from cifar_pipeline.resnet_cifar import resnet110

from torch.nn import DataParallel

MODEL_SAVE_PATH = "models/cifar_resnet110_fixup_0_1"


class Config(ConfigCIFARBase):
    def __init__(self):
        model = resnet110(use_fixup=True, fixup_coeff=0.1)

        super().__init__(model=DataParallel(model), model_save_path=MODEL_SAVE_PATH,
                         epoch_count=100, batch_size=128)
