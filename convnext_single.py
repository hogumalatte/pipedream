"""
Single GPU ConvNeXt Implementation (for validation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.single_gpu_layers import ComplexConv2d, ComplexBatchNorm2d, ComplexLayerNorm

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block for single GPU

    Structure:
    1. Depthwise Conv 7x7
    2. LayerNorm
    3. Pointwise Conv 1x1 (확장)
    4. GELU
    5. Pointwise Conv 1x1 (축소)
    6. Residual
    """
    def __init__(self, dim):
        super().__init__()

        # Depthwise Conv (7x7, groups=dim)
        self.dwconv = ComplexConv2d(
            dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=False
        )

        # LayerNorm
        self.norm = ComplexLayerNorm(dim, eps=1e-6)

        # Pointwise Conv 1: 확장 (dim -> 4*dim)
        self.pwconv1 = ComplexConv2d(
            dim, 4 * dim, kernel_size=1, stride=1, padding=0, bias=False
        )

        # GELU activation
        self.act = nn.GELU()

        # Pointwise Conv 2: 축소 (4*dim -> dim)
        self.pwconv2 = ComplexConv2d(
            4 * dim, dim, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, input_real, input_imag):
        shortcut_r, shortcut_i = input_real, input_imag

        # 1. Depthwise Conv
        x_r, x_i = self.dwconv(input_real, input_imag)

        # 2. LayerNorm
        x_r, x_i = self.norm(x_r, x_i)

        # 3. Pointwise Expansion
        x_r, x_i = self.pwconv1(x_r, x_i)

        # 4. GELU
        x_r, x_i = self.act(x_r), self.act(x_i)

        # 5. Pointwise Projection
        x_r, x_i = self.pwconv2(x_r, x_i)

        # 6. Residual
        x_r = x_r + shortcut_r
        x_i = x_i + shortcut_i

        return x_r, x_i

class SingleConvNeXt(nn.Module):
    """Single GPU ConvNeXt for validation"""

    def __init__(self, depths, dims, num_classes=10):
        super().__init__()

        # Stem: Patchify (4x4 conv with stride 4)
        self.stem = ComplexConv2d(3, dims[0], kernel_size=4, stride=4, padding=0, bias=False)
        self.norm_stem = ComplexLayerNorm(dims[0])

        # 4 stages
        self.stages = nn.ModuleList()

        for i in range(4):
            # Downsampling layer (except for first stage)
            if i > 0:
                downsample = nn.Sequential(
                    ComplexLayerNorm(dims[i-1]),
                    ComplexConv2d(dims[i-1], dims[i], kernel_size=2, stride=2, padding=0, bias=False)
                )
            else:
                downsample = nn.Identity()

            # Blocks
            blocks = nn.ModuleList([ConvNeXtBlock(dims[i]) for _ in range(depths[i])])

            self.stages.append(nn.ModuleDict({
                'downsample': downsample,
                'blocks': blocks
            }))

        # Head
        self.norm_head = ComplexLayerNorm(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dims[-1], num_classes)

    def forward(self, input_real, input_imag):
        # Stem
        x_r, x_i = self.stem(input_real, input_imag)
        x_r, x_i = self.norm_stem(x_r, x_i)

        # Stages
        for stage in self.stages:
            # Downsample
            if isinstance(stage['downsample'], nn.Identity):
                pass
            else:
                x_r, x_i = stage['downsample'][0](x_r, x_i)  # LayerNorm
                x_r, x_i = stage['downsample'][1](x_r, x_i)  # Conv

            # Blocks
            for block in stage['blocks']:
                x_r, x_i = block(x_r, x_i)

        # Head
        x_r, x_i = self.norm_head(x_r, x_i)

        # Global average pooling
        x_r = self.avgpool(x_r)
        x_i = self.avgpool(x_i)

        # Flatten
        x_r = x_r.view(x_r.size(0), -1)
        x_i = x_i.view(x_i.size(0), -1)

        # Classification
        x = x_r
        x = self.fc(x)

        return x

def single_convnext_tiny(num_classes=10):
    """Single GPU ConvNeXt-Tiny"""
    return SingleConvNeXt(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        num_classes=num_classes
    )

def single_convnext_small(num_classes=10):
    """Single GPU ConvNeXt-Small"""
    return SingleConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        num_classes=num_classes
    )

def single_convnext_base(num_classes=10):
    """Single GPU ConvNeXt-Base"""
    return SingleConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        num_classes=num_classes
    )

def single_convnext_large(num_classes=10):
    """Single GPU ConvNeXt-Large"""
    return SingleConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        num_classes=num_classes
    )
