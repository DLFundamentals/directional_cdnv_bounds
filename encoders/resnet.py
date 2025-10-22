from typing import Union, Optional
import torch.nn as nn
from encoders.base import BaseEncoder

class ResNetEncoder(BaseEncoder):
    """
    A wrapper around a ResNet model that extracts
    activations from a specified layer during forward pass
    """
    def __init__(
        self,
        model: nn.Module,
        layer: Union[str, int] = -2,
        dataset: str = 'imagenet',
        width_multiplier: int = 1,
        pretrained: bool = False,
        **kwargs
    ):
        """
        layer: the layer from which to extract the hidden representation
        dataset: the dataset on which the model will be pretrained, either 'imagenet' or 'cifar'
        width_multiplier: the width multiplier for the ResNet model (1, 2, 4, 8, ...)
        pretrained: whether to load pretrained weights (does not work for pretrained weights if width_multiplier > 1)
        """
        super().__init__(model, layer)
        self.width_multiplier = int(width_multiplier)
        self.pretrained = pretrained

        if self.width_multiplier != 1:
            assert not pretrained, 'Pretrained weights not available for wide ResNet.'

        self.create_wider_resnet()
        if 'cifar' in dataset or dataset=='svhn':
            self.modify_for_cifar()
        # for SSL pretraining, we do not need the final fc layer
        self.net.fc = nn.Identity()

    def create_wider_resnet(self):
        if self.width_multiplier == 1:
            return

        # modify the first conv layer
        self.net.conv1 = nn.Conv2d(3, 64 * self.width_multiplier, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.net.bn1 = nn.BatchNorm2d(64 * self.width_multiplier)

        if not self.pretrained:
            self._initialize_weights(self.net.conv1)
            self._initialize_weights(self.net.bn1)
          
        # modify the subsequent layers
        for layer in [self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]:
            for block in layer:
                block.conv1 = self._wider_bottleneck(block.conv1)
                block.bn1 = nn.BatchNorm2d(block.conv1.out_channels)
                block.conv2 = self._wider_bottleneck(block.conv2)
                block.bn2 = nn.BatchNorm2d(block.conv2.out_channels)
                block.conv3 = self._wider_bottleneck(block.conv3)
                block.bn3 = nn.BatchNorm2d(block.conv3.out_channels)

                if block.downsample is not None:
                    block.downsample[0] = self._wider_bottleneck(block.downsample[0])
                    block.downsample[1] = nn.BatchNorm2d(block.downsample[0].out_channels)

    def _wider_bottleneck(self, conv):
        widened = nn.Conv2d(
            conv.in_channels * self.width_multiplier,
            conv.out_channels * self.width_multiplier,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None)
        )
        if not self.pretrained:
            self._initialize_weights(widened)
        del conv
        return widened

    def modify_for_cifar(self):
        # replace the first conv layer to adapt to CIFAR
        self.net.conv1 = nn.Conv2d(3, 64 * self.width_multiplier, 3, 1, 1, bias=False)
        # remove the first max pooling operation
        self.net.maxpool = nn.Identity()

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)