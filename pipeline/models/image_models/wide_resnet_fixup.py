"""
Wide ResNet by Sergey Zagoruyko and Nikos Komodakis
Fixup initialization by Hongyi Zhang, Yann N. Dauphin, Tengyu Ma
Based on code by xternalz and Andy Brock:
https://github.com/xternalz/WideResNet-pytorch
https://github.com/ajbrock/BoilerPlate
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    m = 2

    def __init__(self, in_planes, out_planes, stride, dropout, fixup_l, fixup_coeff):
        super(BasicBlock, self).__init__()

        self._dropout = dropout

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.equalInOut = in_planes == out_planes
        self.conv_res = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_res = not self.equalInOut and self.conv_res or None

        self.multiplicator = nn.Parameter(torch.ones(1,1,1,1))
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1,1,1,1))] * 4)

        k = self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.out_channels
        self.conv1.weight.data.normal_(0, fixup_coeff * fixup_l ** (-1 / (2 * self.m - 2)) * math.sqrt(2. / k))
        self.conv2.weight.data.zero_()

        if self.conv_res is not None:
            k = self.conv_res.kernel_size[0] * self.conv_res.kernel_size[1] * self.conv_res.out_channels
            self.conv_res.weight.data.normal_(0, math.sqrt(2. / k))

    def forward(self, x):
        x_out = self.relu(x + self.biases[0])
        out = self.conv1(x_out) + self.biases[1]
        out = self.relu(out) + self.biases[2]
        if self._dropout > 0:
            out = F.dropout(out, p=self._dropout, training=self.training)
        out = self.multiplicator * self.conv2(out) + self.biases[3]

        if self.equalInOut:
            return torch.add(x, out)

        return torch.add(self.conv_res(x_out), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout, fixup_l, fixup_coeff):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout, fixup_l, fixup_coeff)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout, fixup_l, fixup_coeff):
        layers = []

        for i in range(int(nb_layers)):
            _in_planes = i == 0 and in_planes or out_planes
            _stride = i == 0 and stride or 1
            layers.append(block(_in_planes, out_planes, _stride, dropout=dropout, fixup_l=fixup_l, fixup_coeff=fixup_coeff))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropout=0.0, fixup_coeff=1):
        super(WideResNet, self).__init__()

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        assert (depth - 4) % 6 == 0, "You need to change the number of layers"
        n = (depth - 4) / 6

        block = BasicBlock
        fixup_l = n * 3

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout=dropout,
                                   fixup_l=fixup_l, fixup_coeff=fixup_coeff)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout=dropout,
                                   fixup_l=fixup_l, fixup_coeff=fixup_coeff)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout=dropout,
                                   fixup_l=fixup_l, fixup_coeff=fixup_coeff)

        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.fc.bias.data.zero_()
        self.fc.weight.data.zero_()

        k = self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.out_channels
        self.conv1.weight.data.normal_(0, math.sqrt(2. / k))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.relu(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
