"""
Single GPU RegNet Implementation (for validation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.single_gpu_layers import ComplexConv2d, ComplexBatchNorm2d

class RegNetBlock(nn.Module):
    """RegNet Block for single GPU

    Structure: 1x1 -> 3x3(Group) -> 1x1
    """
    def __init__(self, w_in, w_out, stride, group_width, bottleneck_ratio=1.0):
        super().__init__()
        w_b = int(round(w_out * bottleneck_ratio))  # Bottleneck width
        num_groups = w_b // group_width

        # Conv1 (1x1)
        self.conv1 = ComplexConv2d(w_in, w_b, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = ComplexBatchNorm2d(w_b)

        # Conv2 (3x3 Group)
        self.conv2 = ComplexConv2d(w_b, w_b, kernel_size=3, stride=stride, padding=1,
                                   groups=num_groups, bias=False)
        self.bn2 = ComplexBatchNorm2d(w_b)

        # Conv3 (1x1 Project)
        self.conv3 = ComplexConv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = ComplexBatchNorm2d(w_out)

        self.relu = nn.ReLU()

        # Shortcut
        self.downsample = None
        if stride != 1 or w_in != w_out:
            self.downsample = nn.Sequential(
                ComplexConv2d(w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False),
                ComplexBatchNorm2d(w_out)
            )

    def forward(self, input_real, input_imag):
        identity_r, identity_i = input_real, input_imag

        # Conv1
        out_r, out_i = self.conv1(input_real, input_imag)
        out_r, out_i = self.bn1(out_r, out_i)
        out_r, out_i = self.relu(out_r), self.relu(out_i)

        # Conv2 (Group)
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

class SingleRegNet(nn.Module):
    """Single GPU RegNet for validation"""

    def __init__(self, depths, widths, group_width, num_classes=10, bottleneck_ratio=1.0):
        super().__init__()

        # Stem
        self.stem = ComplexConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = ComplexBatchNorm2d(32)
        self.relu = nn.ReLU()

        # Stages
        self.in_width = 32
        self.stage1 = self._make_stage(widths[0], depths[0], group_width, stride=1,
                                       bottleneck_ratio=bottleneck_ratio)
        self.stage2 = self._make_stage(widths[1], depths[1], group_width, stride=2,
                                       bottleneck_ratio=bottleneck_ratio)
        self.stage3 = self._make_stage(widths[2], depths[2], group_width, stride=2,
                                       bottleneck_ratio=bottleneck_ratio)
        self.stage4 = self._make_stage(widths[3], depths[3], group_width, stride=2,
                                       bottleneck_ratio=bottleneck_ratio)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3], num_classes)

    def _make_stage(self, width, depth, group_width, stride, bottleneck_ratio):
        """Build a stage with multiple blocks"""
        layers = []

        # First block (may downsample)
        layers.append(RegNetBlock(self.in_width, width, stride, group_width, bottleneck_ratio))
        self.in_width = width

        # Remaining blocks
        for _ in range(1, depth):
            layers.append(RegNetBlock(width, width, 1, group_width, bottleneck_ratio))

        return nn.Sequential(*layers)

    def forward(self, input_real, input_imag):
        # Stem
        x_r, x_i = self.stem(input_real, input_imag)
        x_r, x_i = self.bn1(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)

        # Stages
        for block in self.stage1:
            x_r, x_i = block(x_r, x_i)
        for block in self.stage2:
            x_r, x_i = block(x_r, x_i)
        for block in self.stage3:
            x_r, x_i = block(x_r, x_i)
        for block in self.stage4:
            x_r, x_i = block(x_r, x_i)

        # Head
        x_r = self.avgpool(x_r)
        x_i = self.avgpool(x_i)
        x_r = x_r.view(x_r.size(0), -1)
        x_i = x_i.view(x_i.size(0), -1)

        # Combine real and imaginary for classification
        x = x_r
        x = self.fc(x)

        return x

def single_regnet_y_200mf(num_classes=10):
    """Single GPU RegNetY-200MF"""
    return SingleRegNet(
        depths=[1, 1, 4, 1],
        widths=[24, 56, 152, 368],
        group_width=8,
        num_classes=num_classes,
        bottleneck_ratio=1.0
    )

def single_regnet_y_400mf(num_classes=10):
    """Single GPU RegNetY-400MF"""
    return SingleRegNet(
        depths=[1, 3, 6, 1],
        widths=[48, 104, 208, 440],
        group_width=8,
        num_classes=num_classes,
        bottleneck_ratio=1.0
    )
