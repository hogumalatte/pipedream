"""
Single GPU Complex-valued Layers

Common complex-valued layers for single GPU (DDP) implementations.
These layers perform full complex multiplication without tensor parallelism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    """
    Complex-valued 2D Convolution for Single GPU

    Performs complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Stride (default: 1)
        padding: Padding (default: 0)
        groups: Number of groups for grouped convolution (default: 1)
        bias: Whether to use bias (default: False)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

    def forward(self, input_real, input_imag):
        """
        Args:
            input_real: Real part of input [B, C_in, H, W]
            input_imag: Imaginary part of input [B, C_in, H, W]

        Returns:
            tuple: (output_real, output_imag) [B, C_out, H', W']
        """
        # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        real_real = self.conv_r(input_real)  # a * c
        real_imag = self.conv_r(input_imag)  # a * d
        imag_real = self.conv_i(input_real)  # b * c
        imag_imag = self.conv_i(input_imag)  # b * d

        out_real = real_real - imag_imag  # ac - bd
        out_imag = real_imag + imag_real  # ad + bc

        return out_real, out_imag


class ComplexConvTranspose2d(nn.Module):
    """
    Complex-valued 2D Transposed Convolution for Single GPU

    Used for upsampling in encoder-decoder architectures (e.g., U-Net).

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Stride (default: 1)
        padding: Padding (default: 0)
        bias: Whether to use bias (default: False)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.convt_r = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.convt_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, input_real, input_imag):
        """
        Args:
            input_real: Real part of input [B, C_in, H, W]
            input_imag: Imaginary part of input [B, C_in, H, W]

        Returns:
            tuple: (output_real, output_imag) [B, C_out, H', W']
        """
        real_real = self.convt_r(input_real)
        real_imag = self.convt_r(input_imag)
        imag_real = self.convt_i(input_real)
        imag_imag = self.convt_i(input_imag)

        out_real = real_real - imag_imag
        out_imag = real_imag + imag_real

        return out_real, out_imag


class ComplexBatchNorm2d(nn.Module):
    """
    Complex-valued Batch Normalization for Single GPU

    Applies separate batch normalization to real and imaginary parts.
    This is a naive approach that treats real and imaginary parts independently.

    Args:
        num_features: Number of channels
    """
    def __init__(self, num_features):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, input_real, input_imag):
        """
        Args:
            input_real: Real part of input [B, C, H, W]
            input_imag: Imaginary part of input [B, C, H, W]

        Returns:
            tuple: (normed_real, normed_imag)
        """
        return self.bn_r(input_real), self.bn_i(input_imag)


class ComplexLayerNorm(nn.Module):
    """
    Complex-valued Layer Normalization for Single GPU

    Used in ConvNeXt and Transformer-based architectures.
    Applies separate layer normalization to real and imaginary parts.

    Args:
        normalized_shape: Shape to normalize over (typically number of channels)
        eps: Epsilon for numerical stability (default: 1e-6)
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.ln_real = nn.LayerNorm(normalized_shape, eps=eps)
        self.ln_imag = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, input_real, input_imag):
        """
        Args:
            input_real: Real part [B, C, H, W]
            input_imag: Imaginary part [B, C, H, W]

        Returns:
            tuple: (normed_real, normed_imag)

        Note:
            LayerNorm expects [B, H, W, C], so we permute before/after.
            .contiguous() is added to prevent DDP gradient stride mismatch.
        """
        # Permute to channel-last: [B, C, H, W] → [B, H, W, C]
        real = input_real.permute(0, 2, 3, 1)
        imag = input_imag.permute(0, 2, 3, 1)

        # Apply LayerNorm
        real = self.ln_real(real)
        imag = self.ln_imag(imag)

        # Permute back and make contiguous: [B, H, W, C] → [B, C, H, W]
        real = real.permute(0, 3, 1, 2).contiguous()
        imag = imag.permute(0, 3, 1, 2).contiguous()

        return real, imag
