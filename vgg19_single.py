"""
Single GPU VGG-19 Implementation (for validation)
"""

import torch
import torch.nn as nn
from ..layers.single_gpu_layers import ComplexConv2d, ComplexBatchNorm2d

class SingleVGG19(nn.Module):
    """Single GPU VGG-19 for validation"""

    def __init__(self, num_classes=10):
        super().__init__()

        # Block 1
        self.conv1_1 = ComplexConv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = ComplexBatchNorm2d(64)
        self.conv1_2 = ComplexConv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn1_2 = ComplexBatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.conv2_1 = ComplexConv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = ComplexBatchNorm2d(128)
        self.conv2_2 = ComplexConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = ComplexBatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3
        self.conv3_1 = ComplexConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3_1 = ComplexBatchNorm2d(256)
        self.conv3_2 = ComplexConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = ComplexBatchNorm2d(256)
        self.conv3_3 = ComplexConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3_3 = ComplexBatchNorm2d(256)
        self.conv3_4 = ComplexConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3_4 = ComplexBatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4
        self.conv4_1 = ComplexConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4_1 = ComplexBatchNorm2d(512)
        self.conv4_2 = ComplexConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn4_2 = ComplexBatchNorm2d(512)
        self.conv4_3 = ComplexConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn4_3 = ComplexBatchNorm2d(512)
        self.conv4_4 = ComplexConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn4_4 = ComplexBatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Block 5
        self.conv5_1 = ComplexConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5_1 = ComplexBatchNorm2d(512)
        self.conv5_2 = ComplexConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5_2 = ComplexBatchNorm2d(512)
        self.conv5_3 = ComplexConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5_3 = ComplexBatchNorm2d(512)
        self.conv5_4 = ComplexConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5_4 = ComplexBatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, input_real, input_imag):
        # Block 1
        x_r, x_i = self.conv1_1(input_real, input_imag)
        x_r, x_i = self.bn1_1(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.conv1_2(x_r, x_i)
        x_r, x_i = self.bn1_2(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.pool1(x_r), self.pool1(x_i)

        # Block 2
        x_r, x_i = self.conv2_1(x_r, x_i)
        x_r, x_i = self.bn2_1(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.conv2_2(x_r, x_i)
        x_r, x_i = self.bn2_2(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.pool2(x_r), self.pool2(x_i)

        # Block 3
        x_r, x_i = self.conv3_1(x_r, x_i)
        x_r, x_i = self.bn3_1(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.conv3_2(x_r, x_i)
        x_r, x_i = self.bn3_2(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.conv3_3(x_r, x_i)
        x_r, x_i = self.bn3_3(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.conv3_4(x_r, x_i)
        x_r, x_i = self.bn3_4(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.pool3(x_r), self.pool3(x_i)

        # Block 4
        x_r, x_i = self.conv4_1(x_r, x_i)
        x_r, x_i = self.bn4_1(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.conv4_2(x_r, x_i)
        x_r, x_i = self.bn4_2(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.conv4_3(x_r, x_i)
        x_r, x_i = self.bn4_3(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.conv4_4(x_r, x_i)
        x_r, x_i = self.bn4_4(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.pool4(x_r), self.pool4(x_i)

        # Block 5
        x_r, x_i = self.conv5_1(x_r, x_i)
        x_r, x_i = self.bn5_1(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.conv5_2(x_r, x_i)
        x_r, x_i = self.bn5_2(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.conv5_3(x_r, x_i)
        x_r, x_i = self.bn5_3(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.conv5_4(x_r, x_i)
        x_r, x_i = self.bn5_4(x_r, x_i)
        x_r, x_i = self.relu(x_r), self.relu(x_i)
        x_r, x_i = self.pool5(x_r), self.pool5(x_i)

        # Classifier
        x_r = self.avgpool(x_r)
        x_i = self.avgpool(x_i)
        x_r = x_r.view(x_r.size(0), -1)
        x_i = x_i.view(x_i.size(0), -1)
        x = x_r
        x = self.fc(x)
        return x

def single_vgg19(num_classes=10):
    """Single GPU VGG-19"""
    return SingleVGG19(num_classes=num_classes)
