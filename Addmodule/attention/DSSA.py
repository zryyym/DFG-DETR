import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange

from Addmodule.block.HGBlock import AFR


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, nh, N, hd)

        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-2)

        context = k.transpose(-2, -1) @ v  # (B, nh, hd, hd)
        out = q @ context                  # (B, nh, N, hd)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out).transpose(1, 2).contiguous().reshape(B, C, H, W)
        return out


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                 groups=in_channels, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x + residual


class DWConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)

    def forward(self, x):
        return self.conv(x)  # no residual, no ReLU


class DSSA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim // 4

        self.conv3 = DWConv(self.dim, 3)
        self.conv7 = DWConv(self.dim, 7)

        self.attn3 = LinearAttention(self.dim, num_heads)
        self.attn7 = LinearAttention(self.dim, num_heads)
        self.attn_id = LinearAttention(self.dim, num_heads)

        self.fuse_gate = AFR(dim, dim)  # 注意：输入是 dim

        # 可学习的 x_raw 权重（初始抑制）
        self.raw_weight = nn.Parameter(torch.zeros(1))  # scalar


    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.reshape(B, C, H, W)

        x3, x7, x_id, x_raw = x.chunk(4, dim=1)

        y3 = self.attn3(self.conv3(x3))
        y7 = self.attn7(self.conv7(x7))
        y_id = self.attn_id(x_id)

        # 动态加权 x_raw
        x_raw_weighted = x_raw * self.raw_weight.sigmoid()  # [0,1] 缩放

        out = torch.cat([y3, y7, y_id, x_raw_weighted], dim=1)
        fused = self.fuse_gate(out)

        return fused.permute(0, 2, 3, 1).reshape(B, N, C)
