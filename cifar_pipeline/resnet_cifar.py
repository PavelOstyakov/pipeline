import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import math


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1
    m = 2

    def __init__(self, in_planes, planes, stride=1, use_fixup=False, fixup_l=1, fixup_coeff=1):
        super(BasicBlock, self).__init__()
        self._use_fixup = use_fixup
        self._fixup_l = fixup_l
        self._fixup_coeff = fixup_coeff

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

        if use_fixup:
            self.scale = nn.Parameter(torch.ones(1))
            self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(4)])

            k = self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.out_channels
            self.conv1.weight.data.normal_(0, fixup_coeff * fixup_l ** (-1 / (2 * self.m - 2)) * math.sqrt(2. / k))
            self.conv2.weight.data.zero_()

    def forward(self, x):
        if self._use_fixup:
            out = F.relu(self.conv1(x + self.biases[0]) + self.biases[1])
            out = self.scale * self.conv2(out + self.biases[2]) + self.biases[3]
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_fixup=False, fixup_coeff=1):
        super(ResNet, self).__init__()
        self.in_planes = 16

        fixup_l = sum(num_blocks)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16) if not use_fixup else nn.Sequential()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1,
                                       use_fixup=use_fixup, fixup_l=fixup_l, fixup_coeff=fixup_coeff)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2,
                                       use_fixup=use_fixup, fixup_l=fixup_l, fixup_coeff=fixup_coeff)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2,
                                       use_fixup=use_fixup, fixup_l=fixup_l, fixup_coeff=fixup_coeff)
        self.linear = nn.Linear(64, num_classes)

        self.bias1 = nn.Parameter(torch.zeros(1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        if not use_fixup:
            self.apply(_weights_init)
        else:
            self.linear.weight.data.zero_()
            self.linear.bias.data.zero_()

            k = self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.out_channels
            self.conv1.weight.data.normal_(0, math.sqrt(2. / k))

    def _make_layer(self, block, planes, num_blocks, stride, use_fixup, fixup_l, fixup_coeff):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_fixup, fixup_l, fixup_coeff))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)) + self.bias1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out + self.bias2)
        return out


def resnet110(use_fixup=False, fixup_coeff=1):
    return ResNet(BasicBlock, [18, 18, 18], use_fixup=use_fixup, fixup_coeff=fixup_coeff)
