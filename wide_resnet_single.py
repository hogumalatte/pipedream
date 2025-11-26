"""
Single GPU Wide ResNet Implementation (for validation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.single_gpu_layers import ComplexConv2d, ComplexBatchNorm2d

class WideBottleneck(nn.Module):
    """Wide Bottleneck Block for single GPU"""
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, width_multiplier=1, downsample=None):
        super().__init__()

        # Apply width multiplier
        width = int(channels * width_multiplier)

        # Conv1 (1x1)
        self.conv1 = ComplexConv2d(in_channels, width, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = ComplexBatchNorm2d(width)

        # Conv2 (3x3)
        self.conv2 = ComplexConv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = ComplexBatchNorm2d(width)

        # Conv3 (1x1 expansion)
        self.conv3 = ComplexConv2d(width, channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = ComplexBatchNorm2d(channels * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, input_real, input_imag):
        identity_r, identity_i = input_real, input_imag

        # Conv1
        out_r, out_i = self.conv1(input_real, input_imag)
        out_r, out_i = self.bn1(out_r, out_i)
        out_r, out_i = self.relu(out_r), self.relu(out_i)

        # Conv2
        out_r, out_i = self.conv2(out_r, out_i)
        out_r, out_i = self.bn2(out_r, out_i)
        out_r, out_i = self.relu(out_r), self.relu(out_i)

        # Conv3
        out_r, out_i = self.conv3(out_r, out_i)
        out_r, out_i = self.bn3(out_r, out_i)

        # Shortcut
        if self.downsample is not None:
            identity_r, identity_i = self.downsample[0](identity_r, identity_i)
            identity_r, identity_i = self.downsample[1](identity_r, identity_i)

        out_r += identity_r
        out_i += identity_i
        out_r, out_i = self.relu(out_r), self.relu(out_i)

        return out_r, out_i

class SingleWideResNet(nn.Module):
    """Single GPU Wide ResNet for validation"""

    def __init__(self, layers, width_multiplier=1, num_classes=10):
        super().__init__()
        self.width_multiplier = width_multiplier

        # Initial conv
        self.conv1 = ComplexConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = ComplexBatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.in_channels = 64
        self.layer1 = self._make_layer(64, layers[0], stride=1, width_multiplier=width_multiplier)
        self.layer2 = self._make_layer(128, layers[1], stride=2, width_multiplier=width_multiplier)
        self.layer3 = self._make_layer(256, layers[2], stride=2, width_multiplier=width_multiplier)
        self.layer4 = self._make_layer(512, layers[3], stride=2, width_multiplier=width_multiplier)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * WideBottleneck.expansion, num_classes)

    def _make_layer(self, channels, num_blocks, stride, width_multiplier):
        downsample = None

        if stride != 1 or self.in_channels != channels * WideBottleneck.expansion:
            downsample = nn.Sequential(
                ComplexConv2d(self.in_channels, channels * WideBottleneck.expansion,
                             kernel_size=1, stride=stride, padding=0, bias=False),
                ComplexBatchNorm2d(channels * WideBottleneck.expansion)
            )

        layers = []
        layers.append(WideBottleneck(self.in_channels, channels, stride, width_multiplier, downsample))
        self.in_channels = channels * WideBottleneck.expansion

        for _ in range(1, num_blocks):
            layers.append(WideBottleneck(self.in_channels, channels, stride=1, width_multiplier=width_multiplier))

        return nn.Sequential(*layers)

    def forward(self, input_real, input_imag):
        # Initial conv
        x_r, x_i = self.conv1(input_real, input_imag)
        x_r, x_i = self.bn1(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.maxpool(x_r), self.maxpool(x_i)

        # Residual layers
        for block in self.layer1:
            x_r, x_i = block(x_r, x_i)
        for block in self.layer2:
            x_r, x_i = block(x_r, x_i)
        for block in self.layer3:
            x_r, x_i = block(x_r, x_i)
        for block in self.layer4:
            x_r, x_i = block(x_r, x_i)

        # Classification head
        x_r = self.avgpool(x_r)
        x_i = self.avgpool(x_i)
        x_r = x_r.view(x_r.size(0), -1)
        x_i = x_i.view(x_i.size(0), -1)

        x = x_r
        x = self.fc(x)

        return x

def single_wide_resnet50_2(num_classes=10):
    """Single GPU WideResNet-50-2"""
    return SingleWideResNet([3, 4, 6, 3], width_multiplier=2, num_classes=num_classes)

def single_wide_resnet50_4(num_classes=10):
    """Single GPU WideResNet-50-4"""
    return SingleWideResNet([3, 4, 6, 3], width_multiplier=4, num_classes=num_classes)

def single_wide_resnet101_2(num_classes=10):
    """Single GPU WideResNet-101-2"""
    return SingleWideResNet([3, 4, 23, 3], width_multiplier=2, num_classes=num_classes)
