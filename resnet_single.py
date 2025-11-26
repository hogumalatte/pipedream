"""
Single GPU Complex ResNet (DDP Baseline용)

통신 없는 순수 복소수 ResNet
DDP와의 공정한 비교를 위해 사용
"""

import torch
import torch.nn as nn
from ..layers.single import SingleComplexConv2d, SingleComplexBatchNorm2d


class SingleComplexBasicBlock(nn.Module):
    """
    DDP용 BasicBlock (통신 없음)

    구조: 3x3 -> 3x3
    모든 연산이 로컬에서 수행됨
    """
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()

        # Conv1 (3x3)
        self.conv1 = SingleComplexConv2d(
            in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = SingleComplexBatchNorm2d(channels)

        # Conv2 (3x3)
        self.conv2 = SingleComplexConv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = SingleComplexBatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)
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

        # Shortcut
        if self.downsample is not None:
            identity_r, identity_i = self.downsample(input_real, input_imag)

        # Add
        out_r = out_r + identity_r
        out_i = out_i + identity_i

        # Final ReLU
        out_r, out_i = self.relu(out_r), self.relu(out_i)

        return out_r, out_i


class SingleComplexBottleneck(nn.Module):
    """
    DDP용 Bottleneck Block (통신 없음)

    구조: 1x1 -> 3x3 -> 1x1
    모든 연산이 로컬에서 수행됨
    """
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()

        # Conv1 (1x1)
        self.conv1 = SingleComplexConv2d(
            in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = SingleComplexBatchNorm2d(channels)

        # Conv2 (3x3)
        self.conv2 = SingleComplexConv2d(
            channels, channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = SingleComplexBatchNorm2d(channels)

        # Conv3 (1x1)
        self.conv3 = SingleComplexConv2d(
            channels, channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = SingleComplexBatchNorm2d(channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
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
            identity_r, identity_i = self.downsample(input_real, input_imag)

        # Add
        out_r = out_r + identity_r
        out_i = out_i + identity_i

        # Final ReLU
        out_r, out_i = self.relu(out_r), self.relu(out_i)

        return out_r, out_i


class SingleComplexDownsample(nn.Module):
    """DDP용 Downsample module"""
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = SingleComplexConv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.bn = SingleComplexBatchNorm2d(out_channels)

    def forward(self, input_real, input_imag):
        out_r, out_i = self.conv(input_real, input_imag)
        out_r, out_i = self.bn(out_r, out_i)
        return out_r, out_i


class SingleComplexResNet(nn.Module):
    """
    DDP용 Complex ResNet (통신 없음)

    모든 레이어가 SingleComplexConv2d 사용
    PyTorch DDP로 감싸서 사용
    """

    def __init__(self, block_class, layers, num_classes=10):
        """
        Args:
            block_class: Bottleneck 클래스
            layers: 각 stage의 블록 개수 [3, 4, 6, 3] for ResNet-50
            num_classes: 분류 클래스 수
        """
        super().__init__()

        self.in_channels = 64

        # Initial Conv
        self.conv1 = SingleComplexConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = SingleComplexBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self.layer1 = self._make_layer(block_class, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block_class, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block_class, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block_class, 512, layers[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_class.expansion, num_classes)

    def _make_layer(self, block_class, channels, blocks, stride=1):
        """레이어 생성"""
        downsample = None

        if stride != 1 or self.in_channels != channels * block_class.expansion:
            downsample = SingleComplexDownsample(
                self.in_channels,
                channels * block_class.expansion,
                stride
            )

        layers = []
        layers.append(block_class(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block_class.expansion

        for _ in range(1, blocks):
            layers.append(block_class(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, input_real, input_imag):
        """
        Args:
            input_real: 실수부 [B, 3, 32, 32]
            input_imag: 허수부 [B, 3, 32, 32]

        Returns:
            torch.Tensor: logits [B, num_classes]
        """
        # Initial Conv
        x_r, x_i = self.conv1(input_real, input_imag)
        x_r, x_i = self.bn1(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)

        # Residual layers
        for block in self.layer1:
            x_r, x_i = block(x_r, x_i)

        for block in self.layer2:
            x_r, x_i = block(x_r, x_i)

        for block in self.layer3:
            x_r, x_i = block(x_r, x_i)

        for block in self.layer4:
            x_r, x_i = block(x_r, x_i)

        # Magnitude
        magnitude = x_r.pow(2) + x_i.pow(2)

        # Global Average Pooling
        magnitude = self.avgpool(magnitude)
        magnitude = magnitude.view(magnitude.size(0), -1)

        # Classifier
        output = self.fc(magnitude)

        return output


# Factory functions
def single_resnet34(num_classes=10):
    """DDP용 ResNet-34 (통신 없음)"""
    return SingleComplexResNet(SingleComplexBasicBlock, [3, 4, 6, 3], num_classes)


def single_resnet50(num_classes=10):
    """DDP용 ResNet-50 (통신 없음)"""
    return SingleComplexResNet(SingleComplexBottleneck, [3, 4, 6, 3], num_classes)


def single_resnet101(num_classes=10):
    """DDP용 ResNet-101 (통신 없음)"""
    return SingleComplexResNet(SingleComplexBottleneck, [3, 4, 23, 3], num_classes)


def single_resnet152(num_classes=10):
    """DDP용 ResNet-152 (통신 없음)"""
    return SingleComplexResNet(SingleComplexBottleneck, [3, 8, 36, 3], num_classes)
