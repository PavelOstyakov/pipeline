from ..base import ConfigWideResNetBase


class Config(ConfigWideResNetBase):
    def __init__(self):
        super().__init__(num_layers=10000, fixup_coeff=0.1, normalization_type=ConfigWideResNetBase.FIXUP, batch_size=64)
