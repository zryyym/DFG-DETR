import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from Addmodule.attention.CoordinateAttention import CoordinateAttention

class LearnableAffineBlock(nn.Module):
    def __init__(
            self,
            scale_value=1.0,
            bias_value=0.0
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class DynamicRepAffine(nn.Module):
    #理论次优
    def __init__(self, num_channels, groups=8):
        super().__init__()
        self.groups = groups

        # 初始化融合参数占位符
        self.register_buffer('fused_scale', torch.ones(1, num_channels, 1, 1))
        self.register_buffer('fused_bias', torch.zeros(1, num_channels, 1, 1))

        # 动态参数生成器（确保分组兼容）
        mid_chs = max(groups, (num_channels // 16 + groups - 1) // groups * groups)
        self.param_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, mid_chs, 1, groups=groups),
            nn.ReLU(),
            nn.Conv2d(mid_chs, num_channels * 2, 1, groups=groups)
        )

        # 重参数分支
        self.rep_conv = nn.Conv2d(
            num_channels, num_channels,
            kernel_size=3,
            padding=1,
            groups=num_channels,
            bias=False
        )
        self.rep_bn = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        if self.training:
            # 训练模式使用动态参数+重参数分支
            params = self.param_gen(x)
            scale, bias = torch.chunk(params, 2, dim=1)
            return x * scale.sigmoid() + bias + self.rep_bn(self.rep_conv(x))
        else:
            # 推理模式使用预融合参数
            return x * self.fused_scale + self.fused_bias

    def reparameterize(self):
        # 融合重参数分支到主参数
        rep_weight, rep_bias = self._get_rep_params()
        scale = self.param_gen[-1].weight[:, :self.fused_scale.size(1)]  # 维度对齐
        bias = self.param_gen[-1].bias[:self.fused_bias.size(1)]

        # 参数融合计算
        with torch.no_grad():
            self.fused_scale = (torch.sigmoid(scale) + rep_weight).view_as(self.fused_scale)
            self.fused_bias = (bias + rep_bias).view_as(self.fused_bias)

    def _get_rep_params(self):
        # 提取重参数分支的等效仿射参数
        conv_weight = self.rep_conv.weight
        bn_weight = self.rep_bn.weight
        bn_bias = self.rep_bn.bias
        bn_mean = self.rep_bn.running_mean
        bn_var = self.rep_bn.running_var
        eps = 1e-5

        # 融合BN到卷积
        fused_weight = (conv_weight * (bn_weight / torch.sqrt(bn_var + eps))[:, None, None, None])
        fused_bias = bn_bias - bn_weight * bn_mean / torch.sqrt(bn_var + eps)

        # 转换为仿射参数形式
        return fused_weight.sum(dim=(2, 3)), fused_bias


class DynamicAffine(nn.Module):
# 理论最优 shape>=128 and channel <= 256开启，stage1-3开启
    def __init__(self, channels, groups=8):
        super().__init__()
        self.groups = groups

        # 通道自适应参数生成器（仅0.003M参数）
        self.param_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局特征压缩
            nn.Conv2d(channels, max(channels // 16, groups), 1, groups=groups),  # 分组降维
            nn.GELU(),  # 平滑非线性激活
            nn.Conv2d(max(channels // 16, groups), channels * 2, 1, groups=groups)  # 动态生成双参数
        )

        # 残差增强（零初始化保障训练稳定性）
        self.res_scale = nn.Parameter(torch.zeros(1))
        self.res_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 动态生成通道级参数 [B, 2C, 1, 1]
        params = self.param_gen(x)
        scale, bias = torch.chunk(params, 2, dim=1)

        # 仿射变换 + 残差连接
        return x * (scale.sigmoid() + self.res_scale) + (bias + self.res_bias)


class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=True,
            lab=None
    ):
        super().__init__()
        self.out_channels = out_chs
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            if lab is None:
                self.lab = LearnableAffineBlock()
            elif lab == 'rep':
                self.lab = DynamicRepAffine(out_chs)
            elif lab == 'dynamic':
                self.lab = DynamicAffine(out_chs)
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):  # 深度可分离卷积
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
            lab=None,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
            lab=lab,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
            lab=lab,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class HG_Block(nn.Module):
    def __init__(
            self,
            in_chs=64,
            mid_chs=32,
            out_chs=256,
            layer_num=3,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=True,
            agg='ese',
            drop_path=0.,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                    )
                )

        # feature aggregation
        self.total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(
                self.total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            att = None
            aggregation_conv = ConvBNAct(
                self.total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            if agg == 'ese':
                att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class AFR(nn.Module):  # AFR
    def __init__(self, in_chs, out_chs, reduction_ratio=16, groups=8):
        super().__init__()
        self.groups = groups

        # 极简空间注意力 (仅0.05K参数)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),  # 单通道卷积
            nn.Sigmoid()
        )

        # 动态比例预测（压缩中间层）
        self.ratio_predict = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chs, max(4, in_chs // 32), 1),  # 强力压缩
            nn.GELU(),
            nn.Conv2d(max(4, in_chs // 32), 2, 1),
            nn.Sigmoid()
        )

        # 特征变换
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 1),
            nn.BatchNorm2d(out_chs),
            nn.GELU()
        )

        # 共享基础特征（使用深度可分离卷积）
        base_att_chs = out_chs // reduction_ratio
        self.base_att = nn.Sequential(
            # 深度卷积替代分组卷积
            nn.Conv2d(out_chs, out_chs, 3, padding=1, groups=out_chs),
            nn.GELU(),
            # 逐点卷积
            nn.Conv2d(out_chs, base_att_chs, 1),
            nn.GELU()
        )
        # self.base_att = MultiScaleSpatialChannelAttention(out_chs, out_chs, reduction_ratio)

        # 双路注意力（共享权重）
        self.att_branch = nn.Conv2d(base_att_chs, out_chs, 1)  # Local Context Channel Attention 局部上下文感知通道注意力
        self.sigmoid = nn.Sigmoid()

        self.current_epoch_scales = []  # 缓存本 epoch 所有 scale
        self.current_epoch_gates = []  # 缓存本 epoch 所有 gate

    def forward(self, x):
        # 空间注意力增强（极简计算）
        spatial_gray = torch.mean(x, dim=1, keepdim=True)  # Spatial Prior Attention 空间先验注意力
        spatial_att = self.spatial_att(spatial_gray)
        x_enhanced = x * (1 + spatial_att)

        # 动态比例
        ratio_scale, ratio_gate = self.ratio_predict(x_enhanced).unbind(1)

        self.current_epoch_scales.append(ratio_scale.flatten().detach())
        self.current_epoch_gates.append(ratio_gate.flatten().detach())

        # 特征压缩
        x_base = self.conv_reduce(x_enhanced)
        x_reduced = x_base * ratio_scale.unsqueeze(1)

        # 共享特征提取（深度可分离卷积）
        base_feat = self.base_att(x_reduced)

        # 双路注意力（共享权重）
        att = self.sigmoid(self.att_branch(base_feat))

        # 门控输出
        return x_base * (1.0 + att * ratio_gate.unsqueeze(1))

class HG_Block_Gate_test(nn.Module):
    def __init__(
            self,
            in_chs=64,
            mid_chs=32,
            out_chs=256,
            layer_num=3,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=True,
            agg='ese',
            ca=0,
            drop_path=0.,
            gate=0,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                    )
                )

        # feature aggregation
        self.total_chs = in_chs + layer_num * mid_chs

        if gate == 1: #Scale-and-Gate Aggregation(SGA)
            gate_module = AFR(self.total_chs, out_chs)

        else:
            gate_module = ConvBNAct(
                self.total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
        if ca == 1:
            ca_module = CoordinateAttention(out_chs)
        elif ca == 3:
            ca_module = EseModule(out_chs)
        # elif ca == 4:
        #     ca_module = CBAM(out_chs)
        # elif ca == 5:
        #     ca_module = SimAM(out_chs)
        else:
            ca_module = nn.Identity()

        self.aggregation = nn.Sequential(
            gate_module,
            ca_module,
        )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x