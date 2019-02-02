from torchvision.models import resnet
import torch.nn as nn


class ResnetModelFeatureExtractorBase(nn.Module):
    def __init__(self, model, input_channels):
        super().__init__()

        model.fc = nn.Sequential()
        model.avgpool = nn.AdaptiveAvgPool2d(1)

        if input_channels != 3:
            model.conv1 = nn.Conv2d(
                input_channels,
                model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=model.conv1.bias)

        self._model = model

    def forward(self, input):
        return self._model(input)


class Resnet18FeatureExtractor(ResnetModelFeatureExtractorBase):
    NUM_FEATURES = 512

    def __init__(self, pretrained=True, input_channels=3):
        model = resnet.resnet18(pretrained=pretrained)
        super().__init__(
            model=model,
            input_channels=input_channels)


class Resnet34FeatureExtractor(ResnetModelFeatureExtractorBase):
    NUM_FEATURES = 512

    def __init__(self, pretrained=True, input_channels=3):
        model = resnet.resnet34(pretrained=pretrained)
        super().__init__(
            model=model,
            input_channels=input_channels)


class Resnet50FeatureExtractor(ResnetModelFeatureExtractorBase):
    NUM_FEATURES = 2048

    def __init__(self, pretrained=True, input_channels=3):
        model = resnet.resnet50(pretrained=pretrained)
        super().__init__(
            model=model,
            input_channels=input_channels)


class Resnet101FeatureExtractor(ResnetModelFeatureExtractorBase):
    NUM_FEATURES = 2048

    def __init__(self, pretrained=True, input_channels=3):
        model = resnet.resnet101(pretrained=pretrained)
        super().__init__(
            model=model,
            input_channels=input_channels)


class Resnet152FeatureExtractor(ResnetModelFeatureExtractorBase):
    NUM_FEATURES = 2048

    def __init__(self, pretrained=True, input_channels=3):
        model = resnet.resnet152(pretrained=pretrained)
        super().__init__(
            model=model,
            input_channels=input_channels)
