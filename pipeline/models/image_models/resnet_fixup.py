import torch.nn as nn
import math
import torch


class Bottleneck(nn.Module):
    expansion = 4
    m = 3

    def __init__(self, inplanes, planes, stride=1, downsample=None, fixup_l=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.multiplicator = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, 1, 1)) for _ in range(6)])

        k = self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.out_channels
        self.conv1.weight.data.normal_(0, fixup_l ** (-1 / (2 * self.m - 2)) * math.sqrt(2. / k))

        k = self.conv2.kernel_size[0] * self.conv2.kernel_size[1] * self.conv2.out_channels
        self.conv2.weight.data.normal_(0, fixup_l ** (-1 / (2 * self.m - 2)) * math.sqrt(2. / k))
        self.conv3.weight.data.zero_()

        if downsample is not None:
            k = self.downsample.kernel_size[0] * self.downsample.kernel_size[1] * self.downsample.out_channels
            self.downsample.weight.data.normal_(0, math.sqrt(2. / k))

    def forward(self, x):
        residual = x

        out = self.conv1(x + self.biases[0])
        out = self.relu(out + self.biases[1])

        out = self.conv2(out + self.biases[2])
        out = self.relu(out + self.biases[3])

        out = self.multiplicator * self.conv3(out + self.biases[4]) + self.biases[5]

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, input_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        fixup_l = sum(layers)
        self.layer1 = self._make_layer(block, 64, layers[0], fixup_l=fixup_l)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, fixup_l=fixup_l)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, fixup_l=fixup_l)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, fixup_l=fixup_l)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()

        n = self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.out_channels
        self.conv1.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, fixup_l, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride, bias=True)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, fixup_l=fixup_l))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, fixup_l=fixup_l))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
