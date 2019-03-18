import torch
import torch.nn as nn


class VectorCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self._log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        input = self._log_softmax(input)
        loss = -torch.sum(input * target)
        loss = loss / input.shape[0]
        return loss

