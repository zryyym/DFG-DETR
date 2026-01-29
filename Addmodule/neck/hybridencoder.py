import math

from einops import rearrange
import copy

from engine.deim.hybrid_encoder import *
from engine.core import register
from Addmodule.block.HGBlock import LightConvBNAct
import torch
import torch.nn as nn
import torch.nn.functional as F
from Addmodule.block.HGBlock import UltraEfficientGate
from Addmodule.down.DRFD import DRFDv7, DRFDv7_Light_Enhanced, DRFDv7_Light_Enhanced_Optimized, ODR_only_double, ODR_only_dir
from Addmodule.down.wave_down import Down_wt
from Addmodule.attention.MSLA import MSLA, HMSLA, SMSLA, MSLAv2, MSLAv3, DSSA, MSLAv6
from Addmodule.up.Dysample import DySample



class CatFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_list):
        return torch.cat(x_list, dim=1)


class RDWFC(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.num_inputs = num_inputs
        # 初始化为0（训练初期等效普通拼接）
        self.weights = nn.Parameter(torch.zeros(num_inputs, 1, 1, 1))
        self.eps = 1e-6  # 数值稳定性常数

        self._init_weights()

    def _init_weights(self, init_scale=0.1):
        """权重初始化策略"""
        # 推荐使用正态分布小值初始化
        nn.init.normal_(self.weights, mean=0.0, std=init_scale)

    def forward(self, inputs):
        # 1. 计算动态权重（使用平均绝对偏差）
        imp_weights = []
        for feat in inputs:
            # 计算特征图的平均绝对偏差(MAD)
            feat_mean = feat.mean(dim=[1, 2, 3], keepdim=True)
            mad = torch.mean(torch.abs(feat - feat_mean), dim=[1, 2, 3], keepdim=True)
            imp_weights.append(mad + self.eps)

        # 2. 权重归一化
        all_weights = torch.cat(imp_weights, dim=1)
        norm_weights = torch.sigmoid(all_weights)
        norm_weights = torch.split(norm_weights, 1, dim=1)

        # 3. 残差加权（带约束）
        clamp_weights = self.weights.clamp(-3.0, 3.0)
        weighted_feats = []

        for i, feat in enumerate(inputs):
            # 残差连接：原始特征 + 加权特征
            weighted = feat + feat * clamp_weights[i] * norm_weights[i]
            weighted_feats.append(weighted)

        return torch.cat(weighted_feats, dim=1)


class DWFC(nn.Module):
    """超轻量融合门控（替代nn.cat的正确实现）"""

    def __init__(self, num_inputs):
        super().__init__()
        self.num_inputs = num_inputs

        # 极简参数：每个输入一个权重
        self.weights = nn.Parameter(torch.ones(num_inputs, 1, 1, 1))

        # 动态重要性自适应
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs):
        """
        输入: 特征图列表 [feat1, feat2, ...], 每个形状为 [B, C_i, H, W]
        输出: 加权拼接后的特征图 [B, sum(C_i), H, W]
        """
        # 1. 计算动态权重
        imp_weights = []
        for feat in inputs:
            # 特征重要性计算
            imp = self.avg_pool(feat)  # [B, C_i, 1, 1]
            imp = imp.mean(dim=1, keepdim=True)  # [B, 1, 1, 1]
            imp_weights.append(imp)

        # 2. 归一化权重
        all_weights = torch.stack(imp_weights, dim=0)  # [N, B, 1, 1, 1]
        norm_weights = F.softmax(all_weights, dim=0)

        # 3. 应用加权并拼接
        weighted_feats = []
        for i, feat in enumerate(inputs):
            # 应用学习权重和动态重要性
            weighted = feat * self.weights[i] * norm_weights[i]
            weighted_feats.append(weighted)

        # 4. 拼接加权后的特征图
        return torch.cat(weighted_feats, dim=1)  # [B, sum(C_i), H, W]


class OptimizedFusionGate(nn.Module):
    """优化版融合门控（降低推理延迟）"""

    def __init__(self, num_inputs, use_dynamic=True):
        super().__init__()
        self.num_inputs = num_inputs
        self.use_dynamic = use_dynamic  # 动态权重开关

        # 可学习权重（预归一化）
        self.base_weights = nn.Parameter(torch.ones(num_inputs))
        self.base_weights.data = F.softmax(self.base_weights, dim=0)

        # 轻量级动态权重计算（可选）
        if use_dynamic:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = None

    def forward(self, inputs):
        # 静态权重分量（始终使用）
        weighted_feats = []
        for i, feat in enumerate(inputs):
            # 应用基础权重（固定分量）
            weighted = feat * self.base_weights[i]
            weighted_feats.append(weighted)

        # 动态权重分量（可选）
        if self.use_dynamic and self.training:  # 仅在训练时使用动态权重
            dynamic_weights = []
            for feat in inputs:
                imp = self.avg_pool(feat).mean(dim=1, keepdim=True)
                dynamic_weights.append(imp)

            # 归一化动态权重
            dyn_stack = torch.stack(dynamic_weights, dim=0)
            norm_dyn_weights = F.softmax(dyn_stack, dim=0)

            # 应用动态权重
            for i in range(len(weighted_feats)):
                weighted_feats[i] = weighted_feats[i] * norm_dyn_weights[i]

        # 拼接结果
        return torch.cat(weighted_feats, dim=1)


class EfficientFeatureEnhancer(nn.Module):
    """高效特征增强模块 - 优化输入特征质量"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_enhancer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // reduction), channels, 1),
            nn.Sigmoid()
        )

        # 修改空间增强模块，确保正确处理输入通道
        self.spatial_enhancer = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, groups=1),  # 输入通道为1
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道增强
        channel_att = self.channel_enhancer(x)
        x_channel = x * channel_att

        # 空间增强 - 确保使用keepdim=True保持维度
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        spatial_feat = max_pool + avg_pool

        # 空间增强处理
        spatial_att = self.spatial_enhancer(spatial_feat)
        x_spatial = x_channel * spatial_att

        # 残差连接
        return x_spatial + x


class PositionEmbeddingSine(nn.Module):
    """位置编码模块"""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, y_embed, x_embed):
        """前向传播

        参数:
            y_embed: y方向位置嵌入 [B, H, W]
            x_embed: x方向位置嵌入 [B, H, W]
        """
        if self.normalize:
            y_embed = y_embed / (y_embed.max(dim=1, keepdim=True)[0] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed.max(dim=2, keepdim=True)[0] + 1e-6) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=y_embed.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # 正弦/余弦编码
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # 合并x和y方向的位置编码
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            # output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
            output = layer(output)
        if self.norm is not None:
            output = self.norm(output)

        return output

# class TransformerEncoder(nn.Module):
#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super(TransformerEncoder, self).__init__()
#         self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
#         self.num_layers = num_layers
#         self.norm = norm
#
#     def forward(self, src, src_mask=None, pos_embed=None, pos=None) -> torch.Tensor:
#         output = src
#         if pos is not None:
#             sin, cos = pos
#         for layer in self.layers:
#             # output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
#             if pos is not None:
#                 output = layer(output, sin, cos)
#             else:
#                 output = layer(output)
#         if self.norm is not None:
#             output = self.norm(output)
#
#         return output


@register()
class HybridEncoder_v1(HybridEncoder):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='dfine',
                 fuse_module=None,
                 down=None,
                 fuse_block=None,
                 att=None,
                 enhancer=False,
                 neck_type='pafpn',
                 up=None,
                 ):
        super().__init__(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            version=version,
        )
        self.up = up
        self.neck_type = neck_type  # <<< 保存 neck 类型
        assert neck_type in ['fpn', 'pafpn', 'bifpn'], f"Unsupported neck_type: {neck_type}"

        self.att = att
        if self.att == 'msla':
            encoder_layer = MSLA(
                hidden_dim,
                nhead,
            )

            self.encoder = nn.ModuleList([
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in
                range(len(use_encoder_idx))
            ])
        elif self.att == 'dssa':
            encoder_layer = DSSA(
                hidden_dim,
                nhead,
            )

            self.encoder = nn.ModuleList([
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in
                range(len(use_encoder_idx))
            ])
        else:
            encoder_layer = TransformerEncoderLayer(
                hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=enc_act
            )

            self.encoder = nn.ModuleList([
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in
                range(len(use_encoder_idx))
            ])

        self.fpn_fea_fusion_blocks = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        self.up_modules = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            if fuse_module == 'dwfc':
                self.fpn_fea_fusion_blocks.append(DWFC(2))
            elif fuse_module == 'rdwfc':
                self.fpn_fea_fusion_blocks.append(RDWFC(2))
            else:
                self.fpn_fea_fusion_blocks.append(CatFusion())

            if up == 'dy':
                self.up_modules.append(DySample(hidden_dim))
            else:
                self.up_modules.append(nn.Identity())


            self.fpn_blocks.append(RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2,
                                                    round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                                           if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim,
                                                                               round(3 * depth_mult), act=act,
                                                                               expansion=expansion,
                                                                               bottletype=VGGBlock))

        self.downsample_convs = nn.ModuleList()
        self.pan_fea_fusion_blocks = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):

            if down == 'drfdv7':
                self.downsample_convs.append(DRFDv7(hidden_dim))
            elif down == 'enhanced':
                self.downsample_convs.append(DRFDv7_Light_Enhanced(hidden_dim))
            elif down == 'optim':
                self.downsample_convs.append(DRFDv7_Light_Enhanced_Optimized(hidden_dim))
            elif down == 'dir':
                self.downsample_convs.append(ODR_only_dir(hidden_dim))
            elif down == 'double':
                self.downsample_convs.append(ODR_only_double(hidden_dim))
            elif down == 'wt':
                self.downsample_convs.append(Down_wt(hidden_dim, hidden_dim))
            else:
                self.downsample_convs.append(nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, act=act)) \
                                                 if version == 'dfine' else ConvNormLayer_fuse(hidden_dim, hidden_dim,
                                                                                               3, 2, act=act))

            if fuse_module == 'dwfc':
                self.pan_fea_fusion_blocks.append(DWFC(2))
            elif fuse_module == 'rdwfc':
                self.pan_fea_fusion_blocks.append(RDWFC(2))
            else:
                self.pan_fea_fusion_blocks.append(CatFusion())

            if self.neck_type == 'bifpn' and idx < len(in_channels) - 2:
                in_ch = hidden_dim * 3
            else:
                in_ch = hidden_dim * 2
            self.pan_blocks.append(RepNCSPELAN4(in_ch, hidden_dim, hidden_dim * 2,
                                                    round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                                           if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim,
                                                                               round(3 * depth_mult), act=act,
                                                                               expansion=expansion,
                                                                               bottletype=VGGBlock))

        # 轻量级特征增强模块
        # for in_channel in in_channels:
        #     proj = SemanticConsistentProjection(in_channel, hidden_dim)
        #
        #     self.input_proj.append(proj)
        self.enhancer = enhancer
        if enhancer:
            self.feature_enhancers = nn.ModuleList()
            for in_c in in_channels:
                self.feature_enhancers.append(
                    # SpatialChannelJointAttention(hidden_dim)  # 使用输入通道数而不是hidden_dim
                    EfficientFeatureEnhancer(hidden_dim)
                )
        # self.feature_enhancers = nn.ModuleList()
        # for in_c in in_channels:
        #     self.feature_enhancers.append(
        #         EfficientFeatureEnhancer(hidden_dim)  # 使用输入通道数而不是hidden_dim
        #     )

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        # 应用特征增强模块
        # feats = [self.feature_enhancers[i](feat) for i, feat in enumerate(feats)]

        # 投影特征
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # proj_feats = [self.feature_enhancers[i](feat) for i, feat in enumerate(proj_feats)]
        if self.enhancer:
            proj_feats = [self.feature_enhancers[i](feat) for i, feat in enumerate(proj_feats)]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                if self.att == 'la':
                    sin, cos = self.RoPE((h, w))
                    memory: torch.Tensor = self.encoder[i](src_flatten, pos=(sin, cos))
                else:
                    memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # if self.use_psw and self.psw_module is not None:
        #     # 只对 neck 输入特征加权（proj_feats 已是 [P3, P4, P5]）
        #     proj_feats = self.psw_module(proj_feats)


        # >>> 根据 neck_type 选择不同的融合路径 <<<
        if self.neck_type == 'fpn':
            # 仅 top-down 路径，无 bottom-up
            inner_outs = [proj_feats[-1]]
            for idx in range(len(self.in_channels) - 1, 0, -1):
                feat_heigh = inner_outs[0]
                feat_low = proj_feats[idx - 1]
                feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
                inner_outs[0] = feat_heigh
                if self.up is not None:
                    upsample_feat = self.up_modules[len(self.in_channels) - 1 - idx](feat_heigh)
                else:
                    upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
                inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                    self.fpn_fea_fusion_blocks[len(self.in_channels) - 1 - idx]([upsample_feat, feat_low]))
                inner_outs.insert(0, inner_out)
            # FPN 不进行 bottom-up，直接返回 top-down 结果
            return inner_outs

        elif self.neck_type == 'pafpn':
            # 原始 PAFPN：top-down + bottom-up
            inner_outs = [proj_feats[-1]]
            for idx in range(len(self.in_channels) - 1, 0, -1):
                feat_heigh = inner_outs[0]
                feat_low = proj_feats[idx - 1]
                feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
                inner_outs[0] = feat_heigh
                if self.up is not None:
                    upsample_feat = self.up_modules[len(self.in_channels) - 1 - idx](feat_heigh)
                else:
                    upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
                inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                    self.fpn_fea_fusion_blocks[len(self.in_channels) - 1 - idx]([upsample_feat, feat_low]))
                inner_outs.insert(0, inner_out)

            outs = [inner_outs[0]]
            for idx in range(len(self.in_channels) - 1):
                feat_low = outs[-1]
                feat_height = inner_outs[idx + 1]
                downsample_feat = self.downsample_convs[idx](feat_low)
                out = self.pan_blocks[idx](self.pan_fea_fusion_blocks[idx]([downsample_feat, feat_height]))
                outs.append(out)
            return outs

        elif self.neck_type == 'bifpn':
            # BiFPN 风格：双向多次融合（此处为单次双向近似）
            # Top-down path (same as FPN)
            p5 = proj_feats[2]
            p4 = proj_feats[1]
            p3 = proj_feats[0]

            # P5 -> P4
            if self.up is not None:
                p5_up = self.up_modules[0](p5)
            else:
                p5_up = F.interpolate(p5, scale_factor=2., mode='nearest')
            p4_td_fused = self.fpn_fea_fusion_blocks[0]([p5_up, p4])
            p4_td = self.fpn_blocks[0](p4_td_fused)

            # P4_td -> P3
            if self.up is not None:
                p4_up = self.up_modules[1](p4_td)
            else:
                p4_up = F.interpolate(p4_td, scale_factor=2., mode='nearest')
            p3_fused = self.fpn_fea_fusion_blocks[1]([p4_up, p3])
            p3_out = self.fpn_blocks[1](p3_fused)

            # Bottom-up path
            # P3_out -> P4_td
            p3_down = self.downsample_convs[0](p3_out)
            p4_bu_fused = self.pan_fea_fusion_blocks[0]([p3_down, p4_td, p4])  # 注意：融合的是 P4_td
            p4_out = self.pan_blocks[0](p4_bu_fused)

            # P4_out -> P5
            p4_down = self.downsample_convs[1](p4_out)
            p5_fused = self.pan_fea_fusion_blocks[1]([p4_down, p5])
            p5_out = self.pan_blocks[1](p5_fused)

            return [p3_out, p4_out, p5_out]

        else:
            raise ValueError(f"Unknown neck_type: {self.neck_type}")


class SpatialChannelJointAttention(nn.Module):
    """
    Unified Spatial-Channel Joint Attention with non-suppressive property.

    Two modes:
      - 'serial' (default): Spatial → Channel (optimal for defect detection)
      - 'parallel': Concat spatial & channel gates then fuse (for ablation)

    Key features:
      - Non-suppressive: output = x * (1 + gate) >= x
      - Multi-scale spatial context: 3x3(d=1), 3x3(d=2), 5x5(d=1)
      - Dual-pooling channel attention: AvgPool + MaxPool
      - Lightweight: ～18K params (C=256)
    """

    def __init__(self, channels, reduction=8, mode='serial'):
        super().__init__()
        assert mode in ['serial', 'parallel'], "mode must be 'serial' or 'parallel'"
        self.mode = mode
        self.channels = channels
        mid_ch = max(channels // reduction, 16)

        # ==============================
        # 1. 多尺度空间注意力支路（非抑制）
        # ==============================
        self.reduce = nn.Conv2d(channels, mid_ch, kernel_size=1, bias=False)

        self.branch_a = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False)  # 3x3, d=1
        self.branch_b = nn.Conv2d(mid_ch, mid_ch, 3, dilation=2, padding=2, groups=mid_ch, bias=False)  # 3x3, d=2
        self.branch_c = nn.Conv2d(mid_ch, mid_ch, 5, padding=2, groups=mid_ch, bias=False)  # 5x5, d=1

        # 支路权重路由（通道引导）
        self.spatial_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 3, 1),
            nn.Softmax(dim=1)
        )

        # 空间门控生成
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(mid_ch, 1, 1),
            nn.Softplus()
        )

        # ==============================
        # 2. 双池化通道注意力（非抑制）
        # ==============================
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享MLP（减少冗余）
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # ==============================
        # 3. 并行模式专用融合层
        # ==============================
        if mode == 'parallel':
            # 拼接空间门 + 通道权重 → 融合
            self.parallel_fuse = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=1),  # 输入：[B, 2, H, W]
                nn.Softplus()
            )

        self.alpha_spatial = nn.Parameter(torch.tensor(0.1))
        self.alpha_channel = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, C, H, W = x.shape

        # --------------------------------------------------
        # Step 1: 多尺度空间注意力（生成空间门控）
        # --------------------------------------------------
        feat = self.reduce(x)
        a = self.branch_a(feat)
        b = self.branch_b(feat)
        c = self.branch_c(feat)

        w = self.spatial_weights(x)  # [B, 3, 1, 1]
        w0 = w[:, 0:1, :, :]  # [B, 1, 1, 1]
        w1 = w[:, 1:2, :, :]
        w2 = w[:, 2:3, :, :]
        fused_spatial = w0 * a + w1 * b + w2 * c
        spatial_gate = self.spatial_gate(fused_spatial)  # [B, 1, H, W], ≥0

        # 非抑制空间增强
        x_spatial = x * (1.0 + self.alpha_spatial * spatial_gate)

        # --------------------------------------------------
        # Step 2: 双池化通道注意力（生成通道权重）
        # --------------------------------------------------
        avg_feat = self.avg_pool(x).view(B, C)
        max_feat = self.max_pool(x).view(B, C)

        avg_weight = self.sigmoid(self.shared_mlp(avg_feat)).view(B, C, 1, 1)
        max_weight = self.sigmoid(self.shared_mlp(max_feat)).view(B, C, 1, 1)

        # 可学习融合（避免硬编码 0.5）
        alpha = torch.sigmoid(self.alpha_channel)
        channel_gate = alpha * avg_weight + (1 - alpha) * max_weight  # [B, C, 1, 1]

        # 非抑制通道调制
        x_channel = x * (1.0 + channel_gate)

        # --------------------------------------------------
        # Step 3: 融合策略
        # --------------------------------------------------
        if self.mode == 'serial':
            # 最优顺序：先空间增强，再通道调制
            out = x_spatial * (1.0 + channel_gate)
            # 等价于: x * (1 + s_gate) * (1 + c_gate) ≥ x
        else:  # parallel
            # 并行：将空间门（H,W）与通道门（1,1）广播后拼接
            channel_gate_expanded = channel_gate.expand(-1, -1, H, W).mean(dim=1, keepdim=True)  # [B, 1, H, W]
            combined_gates = torch.cat([spatial_gate, channel_gate_expanded], dim=1)  # [B, 2, H, W]
            final_gate = self.parallel_fuse(combined_gates)  # [B, 1, H, W]
            out = x * (1.0 + final_gate)

        return out

class SemanticConsistentProjection(nn.Module):
    """
    轻量（～38K 参数）、多尺度感知、全任务通用的投影模块。
    - 不跨层通信
    - 不生成大矩阵
    - 多尺度信息通过“共享语义先验”隐式注入
    """

    def __init__(self, in_channels, out_channels, shared_token_dim=32):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels

        # 1. 共享的轻量语义令牌（全局可学习先验）
        # 这是关键创新：一个32维向量，编码“通用视觉语义基元”
        self.semantic_token = nn.Parameter(torch.randn(shared_token_dim))
        self.token_proj = nn.Linear(shared_token_dim, out_channels)

        # 2. 通道亲和力预测（极轻量）
        self.affinity = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, shared_token_dim, 1),  # 例如 1024 → 32
            nn.ReLU(inplace=True)
        )

        # 3. 主投影（标准 1x1）
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # 4. 归一化 + 缩放（稳定训练）
        self.norm = nn.BatchNorm2d(out_channels)
        self.gamma = nn.Parameter(torch.zeros(1))  # 初始关闭增强

    def forward(self, x):
        B, C, H, W = x.shape

        # 主投影路径
        proj = self.norm(self.proj(x))  # [B, Cout, H, W]

        # 计算本层特征与共享语义令牌的亲和力
        feat_token = self.affinity(x).squeeze(-1).squeeze(-1)  # [B, 32]
        semantic_prior = self.token_proj(self.semantic_token.unsqueeze(0))  # [1, Cout]

        # 亲和力加权的语义偏置（广播到空间）
        affinity = torch.sigmoid(feat_token @ self.semantic_token)  # [B, 1]
        bias = affinity.view(B, 1, 1, 1) * semantic_prior.view(1, self.out_ch, 1, 1)

        # 自适应增强：proj + gamma * bias
        out = proj + self.gamma * bias

        return out

# class RobustPSW(nn.Module):
#     def __init__(self, hidden_dim=256, num_levels=3, use_high_freq=True):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_levels = num_levels
#         self.use_high_freq = use_high_freq
#
#         # 高频先验提取器（可选）
#         if use_high_freq:
#             self.laplace_conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
#             laplace_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).float()
#             self.laplace_conv.weight.data = laplace_kernel.view(1, 1, 3, 3)
#             self.laplace_conv.requires_grad_(False)
#
#         # 局部描述符：2x2 网格（保留粗略空间结构）
#         self.local_pool = nn.AdaptiveAvgPool2d((2, 2))  # -> [B, C, 2, 2]
#
#         # 投影到统一 token 维度
#         token_dim = 128
#         self.proj = nn.Linear(hidden_dim, token_dim)
#
#         # 轻量 Transformer 编码器（仅 1 层）
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=token_dim, nhead=4, dim_feedforward=256, batch_first=True),
#             num_layers=1
#         )
#
#         # 权重预测头（通道级）
#         self.weight_head = nn.Sequential(
#             nn.Linear(token_dim, hidden_dim),
#             nn.Sigmoid()
#         )
#
#         # 温度参数（可学习，稳定训练）
#         self.temp = nn.Parameter(torch.ones(1) * 2.0)
#
#     def forward(self, features):
#         B, C = features[0].shape[:2]
#         all_tokens = []
#         device = features[0].device
#
#         for i, f in enumerate(features):
#             # 1. 全局 + 局部描述符
#             global_token = F.adaptive_avg_pool2d(f, 1).view(B, C)  # [B, C]
#             local_tokens = self.local_pool(f).view(B, C, 4)  # [B, C, 4]
#             local_tokens = local_tokens.permute(0, 2, 1)  # [B, 4, C]
#
#             # 2. 高频先验（仅对 P3，因小瑕疵主要在低层）
#             if self.use_high_freq and i == 0:  # only for P3
#                 gray = f.mean(dim=1, keepdim=True)  # [B,1,H,W]
#                 lap = self.laplace_conv(gray).abs()
#                 hf_feat = F.adaptive_avg_pool2d(lap, 1).view(B, 1)  # [B,1]
#                 # 将高频强度广播到所有 token
#                 hf_global = global_token * (1 + hf_feat)  # modulation
#                 hf_local = local_tokens * (1 + hf_feat.unsqueeze(-1))
#             else:
#                 hf_global = global_token
#                 hf_local = local_tokens
#
#             # 3. 拼接 tokens: [global; local_1; ...; local_4]
#             tokens = torch.cat([hf_global.unsqueeze(1), hf_local], dim=1)  # [B, 5, C]
#             all_tokens.append(tokens)
#
#         # 4. 拼接所有层级 tokens
#         X = torch.cat(all_tokens, dim=1)  # [B, 15, C]
#         X = self.proj(X)  # [B, 15, token_dim]
#
#         # 5. 跨尺度交互
#         X = self.transformer(X)  # [B, 15, token_dim]
#
#         # 6. 为每个层级预测通道权重
#         weights = []
#         start = 0
#         for _ in range(self.num_levels):
#             level_tokens = X[:, start:start + 5]  # [B, 5, D]
#             level_weight = level_tokens.mean(dim=1)  # [B, D]
#             weight = self.weight_head(level_weight)  # [B, C]
#             weights.append(weight)
#             start += 5
#
#         # 7. 归一化 + 温度缩放（确保数值稳定）
#         weights = torch.stack(weights, dim=1)  # [B, 3, C]
#         weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)  # 归一化
#         weights = weights ** (1.0 / self.temp.clamp(min=0.1))  # 温度软化
#
#         # 8. 加权特征
#         weighted_features = []
#         for i, f in enumerate(features):
#             w = weights[:, i].view(B, C, 1, 1)  # [B, C, 1, 1]
#             weighted_features.append(f * w)
#
#         return weighted_features
#
#
# class LocalAwarePSW(nn.Module):
#     def __init__(self, hidden_dim=256, num_levels=3):
#         super().__init__()
#         self.num_levels = num_levels
#         self.hidden_dim = hidden_dim
#
#         # 高频响应提取器（用于定位潜在瑕疵）
#         self.hf_conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
#         laplace = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).float()
#         self.hf_conv.weight.data = laplace.view(1, 1, 3, 3)
#         self.hf_conv.requires_grad_(False)
#
#         # 融合高频与原始特征
#         self.fuse_convs = nn.ModuleList([
#             nn.Conv2d(hidden_dim + 1, hidden_dim, 1) for _ in range(num_levels)
#         ])
#
#         # 生成空间权重图（sigmoid 归一化）
#         self.weight_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(hidden_dim, 1, 3, padding=1),
#                 nn.Sigmoid()
#             ) for _ in range(num_levels)
#         ])
#
#     def forward(self, features):
#         """
#         features: List[Tensor] of shape [B, C, H_i, W_i], i=0,1,2 (P3, P4, P5)
#         Returns: List[Tensor] of same shapes, each feature map multiplied by its spatial weight
#         """
#         B = features[0].shape[0]
#         device = features[0].device
#
#         # Step 1: 提取高频图（基于 P3）
#         gray_p3 = features[0].mean(dim=1, keepdim=True)  # [B, 1, H3, W3]
#         hf_map_p3 = self.hf_conv(gray_p3).abs()  # [B, 1, H3, W3]
#
#         weighted_features = []
#
#         for i, feat in enumerate(features):
#             h, w = feat.shape[2:]
#
#             # Step 2: 将高频图 resize 到当前尺度
#             if i == 0:
#                 hf_resized = hf_map_p3  # already at P3 resolution
#             else:
#                 hf_resized = F.interpolate(
#                     hf_map_p3, size=(h, w), mode='bilinear', align_corners=False
#                 )  # [B, 1, h, w]
#
#             # Step 3: 融合 + 生成空间权重（仅作用于当前尺度）
#             fused = torch.cat([feat, hf_resized], dim=1)  # [B, C+1, h, w]
#             fused = self.fuse_convs[i](fused)  # [B, C, h, w]
#             weight = self.weight_heads[i](fused)  # [B, 1, h, w]
#
#             # Step 4: 加权（无需跨尺度归一化！）
#             weighted_feat = feat * weight  # [B, C, h, w]
#             weighted_features.append(weighted_feat)
#
#         return weighted_features  # same structure as input
