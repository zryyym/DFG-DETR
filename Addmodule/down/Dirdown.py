import math
import warnings

import torch
import torch.nn as nn
# from mmcv.cnn import build_norm_layer
import torch.nn.functional as F


class ChannelGroupFusion(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.groups = groups
        assert in_channels % groups == 0, "输入通道数必须能被组数整除"
        assert out_channels % groups == 0, "输出通道数必须能被组数整除"

        # 每组独立处理
        self.group_convs = nn.ModuleList(
            [nn.Conv2d(in_channels // groups, out_channels // groups, 1)
             for _ in range(groups)]
        )

    def forward(self, x):
        # 将输入通道分组
        x_group = torch.chunk(x, self.groups, dim=1)

        # 每组独立处理
        out_group = [conv(x_g) for conv, x_g in zip(self.group_convs, x_group)]

        # 拼接结果
        out = torch.cat(out_group, dim=1)

        out = self.channel_shuffle(out, self.groups)

        return out

    def channel_shuffle(self, x, groups):
        """通道Shuffle操作：将通道维度按组重排"""
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        # 1. 将通道维度拆分为 (groups, channels_per_group)
        x = x.view(batch_size, groups, channels_per_group, height, width)  # shape [B, G, C/G, H, W]

        # 2. 转置 (groups, channels_per_group) 维度，实现Shuffle
        x = torch.transpose(x, 1, 2).contiguous()  # shape [B, C/G, G, H, W]

        # 3. 合并通道维度
        x = x.view(batch_size, num_channels, height, width)  # shape [B, C_out, H, W]
        return x


class DirDown(nn.Module): #ODR:Orientation-guided Dynamic Recalibration Downsampling
    def __init__(self, dim, fusion_groups=8):
        super().__init__()
        self.dim = dim
        self.outdim = dim

        # 方向池化（无参操作）
        self.horizontal_pool = nn.AvgPool2d((1, 3), stride=1, padding=(0, 1))
        self.vertical_pool = nn.AvgPool2d((3, 1), stride=1, padding=(1, 0))

        # 超轻量空间注意力 (仅0.1K参数)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),  # 单通道卷积
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # 分支处理
        self.conv_c = nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim)
        self.act_c = nn.GELU()
        self.norm_c = nn.BatchNorm2d(dim)
        self.max_m = nn.MaxPool2d(3, stride=2, padding=1)
        self.norm_m = nn.BatchNorm2d(dim)

        # 融合层
        self.fusion = ChannelGroupFusion(dim * 2, self.outdim, groups=fusion_groups)

    def forward(self, x):
        # 方向特征
        h_feat = self.horizontal_pool(x)
        v_feat = self.vertical_pool(x)
        x_dir = h_feat + v_feat

        # 空间注意力生成 (计算量<0.1GFLOPs)
        spatial_gray = torch.mean(x, dim=1, keepdim=True)  # 通道平均
        spatial_att = self.spatial_att(spatial_gray)

        # 增强方向特征
        x_dir = x_dir * (1 + spatial_att)

        # 分支处理
        conv_branch = self.norm_c(self.act_c(self.conv_c(x_dir)))
        max_branch = self.norm_m(self.max_m(x_dir))

        # 融合
        fused = torch.cat([conv_branch, max_branch], dim=1)
        return self.fusion(fused)