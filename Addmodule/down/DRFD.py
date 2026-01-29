import math
import warnings

import torch
import torch.nn as nn
# from mmcv.cnn import build_norm_layer
import torch.nn.functional as F

from Addmodule.attention.CoordinateAttention import LightCoordAtt
from Addmodule.block.HGBlock import UltraEfficientGate
from engine.backbone.hgnetv2 import ConvBNAct


class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1, groups=4)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)
        x = self.batch_norm(x)
        return x


class DRFD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv_h = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.act_h = nn.GELU()
        self.cut_x = Cut(in_channels=in_channels, out_channels=out_channels)
        self.batch_norm_h = nn.BatchNorm2d(out_channels)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):  # x = [B, C, H, W]
        h = x  # h = [B, H, W, C]
        h = self.conv(h)  # h = [B, 2C, H, W]
        m = h  # m = [B, 2C, H, W]

        # 切片下采样
        x = self.cut_x(x)  # x = [B, 4C, H/2, W/2] --> x = [B, 2C, H/2, W/2]

        # 卷积下采样
        h = self.conv_h(h)  # h = [B, 2C, H/2, W/2]
        h = self.act_h(h)
        h = self.batch_norm_h(h)  # h = [B, 2C, H/2, W/2]

        # 最大池化下采样
        m = self.max_m(m)  # m = [B, C, H/2, W/2]
        m = self.batch_norm_m(m)

        # 拼接
        x = torch.cat([h, x, m], dim=1)  # x = [B, 6C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 2C, H/2, W/2]

        return x



