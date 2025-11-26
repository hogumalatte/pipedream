"""
Single GPU EfficientNet Implementation (for validation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.single_gpu_layers import ComplexConv2d, ComplexBatchNorm2d

class MBConv(nn.Module):
    """MBConv Block for single GPU

    Structure: 1x1(Expand) -> 3x3(Depthwise) -> 1x1(Project)
    """
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size=3):
        super().__init__()
        self.stride = stride
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        hidden_dim = int(in_channels * expand_ratio)

        # Expand (if needed)
        if expand_ratio != 1:
            self.expand_conv = ComplexConv2d(in_channels, hidden_dim, kernel_size=1,
                                            stride=1, padding=0, bias=False)
            self.expand_bn = ComplexBatchNorm2d(hidden_dim)
        else:
            self.expand_conv = None

        # Depthwise
        padding = (kernel_size - 1) // 2
        self.dw_conv = ComplexConv2d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                                    stride=stride, padding=padding, groups=hidden_dim, bias=False)
        self.dw_bn = ComplexBatchNorm2d(hidden_dim)

        # Project
        self.project_conv = ComplexConv2d(hidden_dim, out_channels, kernel_size=1,
                                         stride=1, padding=0, bias=False)
        self.project_bn = ComplexBatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, input_real, input_imag):
        identity_r, identity_i = input_real, input_imag

        # Expand
        if self.expand_conv is not None:
            x_r, x_i = self.expand_conv(input_real, input_imag)
            x_r, x_i = self.expand_bn(x_r, x_i)
            x_r, x_i = self.relu(x_r), self.relu(x_i)
        else:
            x_r, x_i = input_real, input_imag

        # Depthwise
        x_r, x_i = self.dw_conv(x_r, x_i)
        x_r, x_i = self.dw_bn(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)

        # Project
        x_r, x_i = self.project_conv(x_r, x_i)
        x_r, x_i = self.project_bn(x_r, x_i)

        # Skip connection
        if self.use_res_connect:
            x_r += identity_r
            x_i += identity_i

        return x_r, x_i

class SingleEfficientNet(nn.Module):
    """Single GPU EfficientNet for validation"""

    def __init__(self, num_classes=10):
        super().__init__()

        # Stem
        self.stem = ComplexConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = ComplexBatchNorm2d(32)
        self.relu = nn.ReLU()

        # Blocks (simplified EfficientNet-B0 structure)
        self.blocks = nn.ModuleList([
            # Stage 1
            MBConv(32, 16, expand_ratio=1, stride=1, kernel_size=3),

            # Stage 2
            MBConv(16, 24, expand_ratio=6, stride=2, kernel_size=3),
            MBConv(24, 24, expand_ratio=6, stride=1, kernel_size=3),

            # Stage 3
            MBConv(24, 40, expand_ratio=6, stride=2, kernel_size=5),
            MBConv(40, 40, expand_ratio=6, stride=1, kernel_size=5),

            # Stage 4
            MBConv(40, 80, expand_ratio=6, stride=2, kernel_size=3),
            MBConv(80, 80, expand_ratio=6, stride=1, kernel_size=3),
            MBConv(80, 80, expand_ratio=6, stride=1, kernel_size=3),

            # Stage 5
            MBConv(80, 112, expand_ratio=6, stride=1, kernel_size=5),
            MBConv(112, 112, expand_ratio=6, stride=1, kernel_size=5),

            # Stage 6
            MBConv(112, 192, expand_ratio=6, stride=2, kernel_size=5),
            MBConv(192, 192, expand_ratio=6, stride=1, kernel_size=5),

            # Stage 7
            MBConv(192, 320, expand_ratio=6, stride=1, kernel_size=3),
        ])

        # Head
        self.conv_head = ComplexConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_head = ComplexBatchNorm2d(1280)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, input_real, input_imag):
        # Stem
        x_r, x_i = self.stem(input_real, input_imag)
        x_r, x_i = self.bn1(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)

        # Blocks
        for block in self.blocks:
            x_r, x_i = block(x_r, x_i)

        # Head
        x_r, x_i = self.conv_head(x_r, x_i)
        x_r, x_i = self.bn_head(x_r, x_i)
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

def single_efficientnet_b0(num_classes=10):
    """Single GPU EfficientNet-B0"""
    return SingleEfficientNet(num_classes=num_classes)
