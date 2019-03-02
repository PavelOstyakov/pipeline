from ..base import ConfigCIFARBase

from pipeline.models.image_models.wide_resnet_fixup import WideResNet as WideResNetFixup
from pipeline.models.image_models.wide_resnet import WideResNet as WideResNetBatchNorm

from enum import auto
from torch.nn import DataParallel

MODEL_SAVE_PATH = "models/cifar_wideresnet_{}_{}_layers"


class ConfigWideResNetBase(ConfigCIFARBase):
    BATCH_NORM = auto()
    FIXUP = auto()

    def __init__(self, num_layers, fixup_coeff=1, normalization_type=BATCH_NORM, batch_size=128):
        if normalization_type == self.BATCH_NORM:
            model = WideResNetBatchNorm(depth=num_layers, num_classes=10)
            norm_type = "batchnorm"
        else:
            model = WideResNetFixup(depth=num_layers, num_classes=10, fixup_coeff=fixup_coeff)
            norm_type = "fixup_coeff_{}".format(fixup_coeff)

        super().__init__(model=DataParallel(model), model_save_path=MODEL_SAVE_PATH.format(norm_type, num_layers),
                         epoch_count=1, batch_size=batch_size)
