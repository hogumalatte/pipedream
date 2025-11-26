"""
Single GPU Complex Layers (DDP Baseline용)

통신이 전혀 없는 순수 복소수 연산 레이어
DDP와의 공정한 비교를 위해 사용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleComplexConv2d(nn.Module):
    """
    단일 GPU용 복소수 Conv2d (통신 없음)

    DDP 비교를 위한 순수 연산 레이어
    모든 연산이 로컬에서 수행됨
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, groups=1, bias=True):
        """
        Args:
            in_channels: 입력 채널 수
            out_channels: 출력 채널 수
            kernel_size: 커널 크기
            stride: 스트라이드
            padding: 패딩
            groups: 그룹 수
            bias: 편향 사용 여부
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # 복소수 가중치
        self.weight_real = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.weight_imag = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        )

        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_channels))
            self.bias_imag = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Kaiming 초기화"""
        nn.init.kaiming_normal_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.weight_imag, a=math.sqrt(5))

        if self.bias_real is not None:
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)

    def forward(self, input_real, input_imag):
        """
        복소수 곱셈: (a + bi)(c + di) = (ac - bd) + (ad + bc)i

        Args:
            input_real: 입력의 실수부 [B, C, H, W]
            input_imag: 입력의 허수부 [B, C, H, W]

        Returns:
            tuple: (output_real, output_imag)
        """
        # Real part: IR*WR - II*WI
        rr = F.conv2d(input_real, self.weight_real,
                     stride=self.stride, padding=self.padding, groups=self.groups)
        ii = F.conv2d(input_imag, self.weight_imag,
                     stride=self.stride, padding=self.padding, groups=self.groups)
        out_real = rr - ii

        # Imag part: IR*WI + II*WR
        ri = F.conv2d(input_real, self.weight_imag,
                     stride=self.stride, padding=self.padding, groups=self.groups)
        ir = F.conv2d(input_imag, self.weight_real,
                     stride=self.stride, padding=self.padding, groups=self.groups)
        out_imag = ri + ir

        # Bias
        if self.bias_real is not None:
            out_real = out_real + self.bias_real.view(1, -1, 1, 1)
            out_imag = out_imag + self.bias_imag.view(1, -1, 1, 1)

        return out_real, out_imag


class SingleComplexBatchNorm2d(nn.Module):
    """단일 GPU용 복소수 BatchNorm"""

    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)

    def forward(self, input_real, input_imag):
        return self.bn_real(input_real), self.bn_imag(input_imag)
