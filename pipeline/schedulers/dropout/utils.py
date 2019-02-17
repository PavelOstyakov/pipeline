import abc
from torch.nn.modules.dropout import _DropoutNd


def set_dropout_probability(module, probability):
    if isinstance(module, _DropoutNd):
        module.p = probability
        return

    for child in module.children():
        set_dropout_probability(child, probability)
