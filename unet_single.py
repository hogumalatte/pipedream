"""
Single GPU U-Net Implementation (for DDP validation)

U-Net with complex-valued operations on single GPU
Used for DDP comparison with TP strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.single_gpu_layers import ComplexConv2d, ComplexBatchNorm2d, ComplexConvTranspose2d

class DoubleConv(nn.Module):
    """
    Double Convolution Block: Conv-BN-ReLU-Conv-BN-ReLU

    Basic building block for U-Net
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        # Conv1
        self.conv1 = ComplexConv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = ComplexBatchNorm2d(mid_channels)

        # Conv2
        self.conv2 = ComplexConv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = ComplexBatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, input_real, input_imag):
        # Conv1 + BN + ReLU
        out_real, out_imag = self.conv1(input_real, input_imag)
        out_real, out_imag = self.bn1(out_real, out_imag)
        out_real, out_imag = self.relu(out_real), self.relu(out_imag)

        # Conv2 + BN + ReLU
        out_real, out_imag = self.conv2(out_real, out_imag)
        out_real, out_imag = self.bn2(out_real, out_imag)
        out_real, out_imag = self.relu(out_real), self.relu(out_imag)

        return out_real, out_imag

class Down(nn.Module):
    """Downsampling Block: MaxPool + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, input_real, input_imag):
        # MaxPool (apply independently to real and imaginary)
        pooled_real = self.maxpool(input_real)
        pooled_imag = self.maxpool(input_imag)

        # DoubleConv
        out_real, out_imag = self.conv(pooled_real, pooled_imag)

        return out_real, out_imag

class Up(nn.Module):
    """Upsampling Block: ConvTranspose + DoubleConv with skip connection"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        self.bilinear = bilinear

        # Upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # ConvTranspose2d: reduce channels by half while upsampling
            self.up = ComplexConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # After skip connection concat, channels become in_channels
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1_real, x1_imag, x2_real, x2_imag):
        """
        Args:
            x1_real, x1_imag: Decoder path input (to be upsampled)
            x2_real, x2_imag: Encoder path skip connection (to be concatenated)
        """
        # Upsampling
        if self.bilinear:
            up_real = self.up(x1_real)
            up_imag = self.up(x1_imag)
        else:
            up_real, up_imag = self.up(x1_real, x1_imag)

        # Handle size mismatch with padding
        diff_h = x2_real.size(2) - up_real.size(2)
        diff_w = x2_real.size(3) - up_real.size(3)

        up_real = F.pad(up_real, [diff_w // 2, diff_w - diff_w // 2,
                                   diff_h // 2, diff_h - diff_h // 2])
        up_imag = F.pad(up_imag, [diff_w // 2, diff_w - diff_w // 2,
                                   diff_h // 2, diff_h - diff_h // 2])

        # Concatenate skip connection
        cat_real = torch.cat([x2_real, up_real], dim=1)
        cat_imag = torch.cat([x2_imag, up_imag], dim=1)

        # DoubleConv
        out_real, out_imag = self.conv(cat_real, cat_imag)

        return out_real, out_imag

class SingleUNet(nn.Module):
    """
    Single GPU U-Net for DDP

    Standard U-Net architecture with complex-valued operations

    Args:
        in_channels: Input channels (e.g., 3 for RGB FFT)
        out_channels: Output channels (e.g., 10 for CIFAR-10)
        features: Initial feature count (default: 64)
        bilinear: Use bilinear upsampling instead of ConvTranspose
    """
    def __init__(self, in_channels=3, out_channels=10, features=64, bilinear=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Factor for bilinear
        factor = 2 if bilinear else 1

        # Initial convolution
        self.inc = DoubleConv(in_channels, features)

        # Encoder (Downsampling)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)

        # Bottleneck
        self.down4 = Down(features * 8, features * 16 // factor)

        # Decoder (Upsampling)
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)

        # Output convolution (1x1)
        self.outc = ComplexConv2d(features, out_channels, kernel_size=1, padding=0)

    def forward(self, input_real, input_imag):
        """
        Args:
            input_real: Real part [B, C, H, W]
            input_imag: Imaginary part [B, C, H, W]

        Returns:
            Output tensor (magnitude) [B, num_classes]
        """
        # Encoder
        x1_real, x1_imag = self.inc(input_real, input_imag)
        x2_real, x2_imag = self.down1(x1_real, x1_imag)
        x3_real, x3_imag = self.down2(x2_real, x2_imag)
        x4_real, x4_imag = self.down3(x3_real, x3_imag)

        # Bottleneck
        x5_real, x5_imag = self.down4(x4_real, x4_imag)

        # Decoder (with skip connections)
        x_real, x_imag = self.up1(x5_real, x5_imag, x4_real, x4_imag)
        x_real, x_imag = self.up2(x_real, x_imag, x3_real, x3_imag)
        x_real, x_imag = self.up3(x_real, x_imag, x2_real, x2_imag)
        x_real, x_imag = self.up4(x_real, x_imag, x1_real, x1_imag)

        # Output
        out_real, out_imag = self.outc(x_real, x_imag)

        # Magnitude for classification
        out = torch.sqrt(out_real**2 + out_imag**2).mean(dim=(2, 3))

        return out

def single_unet(in_channels=3, out_channels=10, features=32, bilinear=False):
    """
    Single GPU U-Net factory function

    Args:
        in_channels: Input channels (default: 3)
        out_channels: Output channels (default: 10 for CIFAR-10)
        features: Initial feature count (default: 32 for lightweight)
        bilinear: Use bilinear upsampling (default: False)

    Returns:
        SingleUNet model
    """
    return SingleUNet(in_channels, out_channels, features, bilinear)
