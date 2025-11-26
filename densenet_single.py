"""
Single GPU DenseNet Implementation (for validation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.single_gpu_layers import ComplexConv2d, ComplexBatchNorm2d

class DenseLayer(nn.Module):
    """Dense Layer for single GPU

    Structure: BN -> ReLU -> 1x1 Conv -> BN -> ReLU -> 3x3 Conv
    """
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        self.bn1 = ComplexBatchNorm2d(in_channels)
        self.conv1 = ComplexConv2d(in_channels, bn_size * growth_rate,
                                   kernel_size=1, stride=1, padding=0, bias=False)

        self.bn2 = ComplexBatchNorm2d(bn_size * growth_rate)
        self.conv2 = ComplexConv2d(bn_size * growth_rate, growth_rate,
                                   kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        """
        Args:
            inputs: List of (real, imag) tuples from all previous layers
        """
        # Concatenate all previous outputs
        if isinstance(inputs, list):
            concat_r = torch.cat([r for r, i in inputs], dim=1)
            concat_i = torch.cat([i for r, i in inputs], dim=1)
        else:
            concat_r, concat_i = inputs

        # Conv1 (1x1)
        out_r, out_i = self.bn1(concat_r, concat_i)
        out_r, out_i = self.relu(out_r), self.relu(out_i)
        out_r, out_i = self.conv1(out_r, out_i)

        # Conv2 (3x3)
        out_r, out_i = self.bn2(out_r, out_i)
        out_r, out_i = self.relu(out_r), self.relu(out_i)
        out_r, out_i = self.conv2(out_r, out_i)

        return out_r, out_i

class DenseBlock(nn.Module):
    """Dense Block for single GPU"""

    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(DenseLayer(layer_in_channels, growth_rate, bn_size))

    def forward(self, input_real, input_imag):
        features = [(input_real, input_imag)]

        for layer in self.layers:
            new_r, new_i = layer(features)
            features.append((new_r, new_i))

        # Concatenate all features
        out_r = torch.cat([r for r, i in features], dim=1)
        out_i = torch.cat([i for r, i in features], dim=1)

        return out_r, out_i

class Transition(nn.Module):
    """Transition Layer for single GPU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = ComplexBatchNorm2d(in_channels)
        self.conv = ComplexConv2d(in_channels, out_channels,
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, input_real, input_imag):
        out_r, out_i = self.bn(input_real, input_imag)
        out_r, out_i = self.relu(out_r), self.relu(out_i)
        out_r, out_i = self.conv(out_r, out_i)
        out_r, out_i = self.pool(out_r), self.pool(out_i)

        return out_r, out_i

class SingleDenseNet(nn.Module):
    """Single GPU DenseNet for validation"""

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, num_classes=10):
        super().__init__()

        # Stem
        self.stem = ComplexConv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = ComplexBatchNorm2d(num_init_features)
        self.relu = nn.ReLU()

        # Dense Blocks
        self.features = nn.ModuleList()
        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            # Add dense block
            block = DenseBlock(num_layers, num_features, growth_rate, bn_size)
            self.features.append(block)
            num_features = num_features + num_layers * growth_rate

            # Add transition layer (except after last block)
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.append(trans)
                num_features = num_features // 2

        # Final BN
        self.bn_final = ComplexBatchNorm2d(num_features)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, input_real, input_imag):
        # Stem
        x_r, x_i = self.stem(input_real, input_imag)
        x_r, x_i = self.bn1(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)

        # Dense blocks and transitions
        for layer in self.features:
            x_r, x_i = layer(x_r, x_i)

        # Final BN + ReLU
        x_r, x_i = self.bn_final(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)

        # Pooling
        x_r = self.avgpool(x_r)
        x_i = self.avgpool(x_i)
        x_r = x_r.view(x_r.size(0), -1)
        x_i = x_i.view(x_i.size(0), -1)

        # Classification
        x = x_r
        x = self.fc(x)

        return x

def single_densenet121(num_classes=10):
    """Single GPU DenseNet-121"""
    return SingleDenseNet(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        num_classes=num_classes
    )

def single_densenet169(num_classes=10):
    """Single GPU DenseNet-169"""
    return SingleDenseNet(
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        num_init_features=64,
        num_classes=num_classes
    )
