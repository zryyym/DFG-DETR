import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6  # 使用ReLU6近似sigmoid，减少计算量


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)  # 轻量化非线性激活函数


class FabricTextureAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.in_channels = in_channels
        mip = max(8, in_channels // reduction)

        # --------------------------
        # 原CA：坐标编码（问题根源）
        # self.pool_h = lambda x: torch.mean(x, dim=3, keepdim=True)  # x轴坐标
        # self.pool_w = lambda x: torch.mean(x, dim=2, keepdim=True)  # y轴坐标
        # --------------------------

        # 改进1：纹理方差编码（区分背景/瑕疵）
        self.local_var = nn.AvgPool2d(3, stride=1, padding=1)  # 局部方差计算（3x3窗口）
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        # 保留CA的后续结构（最小改动）
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape

        # --------------------------
        x_local = F.avg_pool2d(x, 3, stride=1, padding=1)  # 局部池化（3x3窗口，保留细节）
        x_global = F.adaptive_avg_pool2d(x, (h, w))  # 全局池化（保留全局上下文）
        x_fused = x_local * 0.7 + x_global * 0.3  # 加权融合（侧重局部）
        # --------------------------

        # 改进1：纹理方差特征（替代坐标特征）
        x_mean = F.avg_pool2d(x_fused, 3, stride=1, padding=1)  # 局部均值
        x_var = self.local_var((x_fused - x_mean) ** 2)  # 局部方差（衡量纹理突变）
        y = x_var  # [n,c,h,w]（直接用方差作为特征，无需拼接）

        # 后续流程复用CA（仅修改输入特征y）
        y = self.act(self.bn(self.conv1(y)))  # [n,mip,h,w]
        x_h = self.sigmoid(self.conv_h(y))  # [n,c,h,w]（水平注意力）
        x_w = self.sigmoid(self.conv_w(y))  # [n,c,h,w]（垂直注意力）

        return identity * x_h * x_w


class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.in_channels = in_channels
        mip = max(8, in_channels // reduction)

        # 替换为ONNX兼容的池化方法
        self.pool_h = lambda x: torch.mean(x, dim=3, keepdim=True)  # (H,1)维度池化
        self.pool_w = lambda x: torch.mean(x, dim=2, keepdim=True)  # (1,W)维度池化

        # 共享卷积层
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()  # 使用标准Hardswish激活

        # 注意力生成
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x

        # 获取维度信息
        n, c, h, w = x.shape

        # 水平方向编码 (N,C,H,1)
        x_h = self.pool_h(x)

        # 垂直方向编码 (N,C,1,W)
        x_w = self.pool_w(x)

        # 拼接并处理特征
        y = torch.cat([x_h, x_w.permute(0, 1, 3, 2)], dim=2)  # (N,C,H+W,1)
        y = self.act(self.bn(self.conv1(y)))

        # 分离特征
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 恢复维度

        # 生成注意力图
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


# class HardSigmoid(nn.Module):
#     def forward(self, x):
#         return torch.clamp(x + 3, 0, 6) / 6.0
#
# def fuse_conv_bn(conv, bn):
#     """Fuse Conv2d and BatchNorm2d into a single Conv2d with bias."""
#     fused_conv = nn.Conv2d(
#         conv.in_channels,
#         conv.out_channels,
#         conv.kernel_size,
#         conv.stride,
#         conv.padding,
#         conv.dilation,
#         conv.groups,
#         bias=True
#     )
#     # Compute fused weight
#     w_conv = conv.weight.clone().view(conv.out_channels, -1)
#     w_bn = torch.diag(bn.weight / torch.sqrt(bn.running_var + bn.eps))
#     fused_conv.weight.data = torch.mm(w_bn, w_conv).view(fused_conv.weight.shape)
#     # Compute fused bias
#     if conv.bias is not None:
#         b_conv = conv.bias
#     else:
#         b_conv = torch.zeros(conv.out_channels, device=conv.weight.device)
#     b_bn = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)
#     fused_conv.bias.data = torch.mm(w_bn, b_conv.unsqueeze(1)).squeeze(1) + b_bn
#     return fused_conv
#
# # ==============================================================================
# # CoordinateAttention: 支持 fuse()
# # ==============================================================================
# class CoordinateAttention(nn.Module):
#     def __init__(self, in_channels, reduction=32):
#         super().__init__()
#         self.in_channels = in_channels
#         mip = max(8, in_channels // reduction)
#
#         self.pool_h = lambda x: x.mean(dim=3, keepdim=True)
#         self.pool_w = lambda x: x.mean(dim=2, keepdim=True)
#
#         # Training modules
#         self.conv1 = nn.Conv2d(in_channels, mip, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = nn.Hardswish()
#         self.conv_h = nn.Conv2d(mip, in_channels, 1, bias=True)
#         self.conv_w = nn.Conv2d(mip, in_channels, 1, bias=True)
#         self.sigmoid = nn.Sigmoid()
#
#         # Fused flag
#         self._is_fused = False
#
#     def forward(self, x):
#         identity = x
#         n, c, h, w = x.shape
#
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x)
#         y = torch.cat([x_h, x_w.permute(0, 1, 3, 2)], dim=2)
#
#         if not self._is_fused:
#             y = self.act(self.bn1(self.conv1(y)))
#         else:
#             y = self.act(self.conv1(y))  # fused conv1 has bias
#
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         a_h = self.sigmoid(self.conv_h(x_h))
#         a_w = self.sigmoid(self.conv_w(x_w))
#
#         return identity * a_h * a_w
#
#     def fuse(self):
#         if self._is_fused:
#             return self
#         # Fuse conv1 + bn1
#         self.conv1 = fuse_conv_bn(self.conv1, self.bn1)
#         delattr(self, 'bn1')
#         # Replace sigmoid with hard_sigmoid
#         self.sigmoid = HardSigmoid()
#         self._is_fused = True
#         return self



class LightweightComplementaryAttention(nn.Module):
    def __init__(self, in_channels, modulation=False):
        super().__init__()
        self.modulation = modulation

        if modulation:
            # 极轻量调制：1x1 conv + sigmoid，参数极少
            self.mod_avg = nn.Sequential(
                nn.Conv2d(1, 1, 1),
                nn.Sigmoid()
            )
            self.mod_max = nn.Sequential(
                nn.Conv2d(1, 1, 1),
                nn.Sigmoid()
            )
        else:
            self.mod_avg = self.mod_max = nn.Identity()

        # 融合卷积（1x1）
        self.fuse = nn.Sequential(
            nn.Conv2d(2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 全局统计（保持高分辨率）
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B,1,H,W)

        if self.modulation:
            # 轻量调制：让 avg 调制 max，max 调制 avg（交叉）
            mod_avg = self.mod_avg(avg_out)
            mod_max = self.mod_max(max_out)
            out = torch.cat([mod_avg * max_out, mod_max * avg_out], dim=1)
        else:
            out = torch.cat([avg_out, max_out], dim=1)

        # 生成注意力掩模
        att = self.fuse(out)  # (B,1,H,W)
        return x * att


class ResidualGateLCA_Modulated(nn.Module):
    """
    方案2 增强版：带调制的残差门控 LCA
    - 保留 avg/max 的交叉调制（增强非线性）
    - 但最终输出仍为 x * (1 + alpha * att)，保证非抑制
    """

    def __init__(self, in_channels, alpha_init=0.5, learnable_alpha=True):
        super().__init__()
        self.learnable_alpha = learnable_alpha

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.register_buffer('alpha', torch.tensor(alpha_init))

        # 调制分支（极轻量）
        self.mod_avg = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid()
        )
        self.mod_max = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid()
        )

        # 融合（注意：输入仍是2通道）
        self.fuse = nn.Sequential(
            nn.Conv2d(2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 调制：交叉门控（增强非线性）
        mod_avg = self.mod_avg(avg_out)  # (B,1,H,W)
        mod_max = self.mod_max(max_out)  # (B,1,H,W)
        gated_avg = mod_max * avg_out  # max 调制 avg
        gated_max = mod_avg * max_out  # avg 调制 max

        att = self.fuse(torch.cat([gated_avg, gated_max], dim=1))  # (B,1,H,W)

        # 残差门控：绝不抑制
        alpha = torch.clamp(self.alpha, 0.0, 2.0)
        return x * (1.0 + alpha * att)


class DCAPlusPlus(nn.Module):
    def __init__(self, channels, reduction=16, num_dirs=4, max_size=1024):
        super().__init__()
        self.channels = channels
        self.num_dirs = num_dirs
        reduced_ch = max(channels // reduction, 8)

        # ✅ 修正：pos_embed_h 是 [1,1,max_H,1]，pos_embed_w 是 [1,1,1,max_W]
        self.pos_embed_h = nn.Parameter(torch.randn(1, 1, max_size, 1) * 0.02)
        self.pos_embed_w = nn.Parameter(torch.randn(1, 1, 1, max_size) * 0.02)

        self.dir_proj = nn.Sequential(
            nn.Conv2d(channels, reduced_ch, 1, bias=False),
            nn.BatchNorm2d(reduced_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, num_dirs, 1, bias=False)
        )

        self.fuse_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_dirs, num_dirs, 1),
            nn.Sigmoid()
        )
        self.conv_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # ✅ 正确切片
        pos_h = self.pos_embed_h[:, :, :H, :].expand(B, 1, H, W)  # [B,1,H,W]
        pos_w = self.pos_embed_w[:, :, :, :W].expand(B, 1, H, W)  # [B,1,H,W]

        x_pos = torch.cat([x.mean(dim=1, keepdim=True), pos_h, pos_w], dim=1)

        dir_map = self.dir_proj(x)  # [B, 4, H, W]
        dir_weights = dir_map.softmax(dim=1)

        x_exp = x.unsqueeze(1)           # [B,1,C,H,W]
        dir_weights_exp = dir_weights.unsqueeze(2)  # [B,4,1,H,W]
        x_dir = x_exp * dir_weights_exp  # [B,4,C,H,W]

        gate = self.fuse_gate(dir_map)   # [B,4,1,1]
        x_fused = (x_dir * gate.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

        return self.conv_out(x_fused) + x


class DALA(nn.Module):
    def __init__(self, in_channels=None):  # in_channels 仅为了接口兼容
        super().__init__()
        # 使用 nn.Conv2d 实现固定方向滤波器（冻结权重）
        self.dir_conv = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            padding=1,
            bias=False  # 无偏置，纯滤波
        )

        # 定义 4 个方向的 Sobel-like 滤波器
        filters = torch.FloatTensor([
            [[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]],      # vertical
            [[[-1,-1,-1], [ 0, 0, 0], [ 1, 1, 1]]],      # horizontal
            [[[-1,-1, 0], [-1, 0, 1], [ 0, 1, 1]]],      # diag ↘
            [[[ 0, 1, 1], [-1, 0, 1], [-1,-1, 0]]],      # diag ↙
        ])  # shape: [4, 1, 3, 3]

        # 将固定滤波器加载到卷积层并冻结
        with torch.no_grad():
            self.dir_conv.weight.copy_(filters)
        self.dir_conv.weight.requires_grad = False  # 冻结，不参与训练

        # 轻量门控（1 可学习参数）
        self.gate = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.gate.bias, -1.0)  # 初始抑制

    def forward(self, x):
        B, C, H, W = x.shape

        # Step 1: 全局方向响应（单通道灰度图）
        gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        dir_resp = self.dir_conv(gray)      # [B, 4, H, W] ← 使用 nn.Conv2d
        global_dir = dir_resp.abs().max(dim=1, keepdim=True).values  # [B, 1, H, W]

        # Step 2: 局部方向异常
        mu_h = x.mean(dim=3, keepdim=True)   # [B, C, H, 1]
        mu_w = x.mean(dim=2, keepdim=True)   # [B, C, 1, W]
        local_anomaly = (x - mu_h).abs().mean(dim=1, keepdim=True) + \
                        (x - mu_w).abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Step 3: 融合 + 门控
        fusion = global_dir * local_anomaly
        attention = h_sigmoid()(self.gate(fusion))

        return x * attention


class NA_LDA(nn.Module):
    def __init__(self, channels, reduction=32, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # 可学习方向权重（全局）
        self.dir_weight_h = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.dir_weight_w = nn.Parameter(torch.ones(1, channels, 1, 1))

        # 轻量门控
        self.gate = nn.Conv2d(1, 1, 1, bias=True)
        nn.init.constant_(self.gate.bias, -1.0)

        # 用于局部协方差的邻域聚合（无参数！）
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        B, C, H, W = x.shape

        # === 全局方向（不变）===
        weight_w = torch.softmax(self.dir_weight_w, dim=1)
        global_h = (x * weight_w).mean(dim=3, keepdim=True)
        weight_h = torch.softmax(self.dir_weight_h, dim=1)
        global_w = (x * weight_h).mean(dim=2, keepdim=True)
        global_dir = global_h.expand(-1,-1,-1,W) + global_w.expand(-1,-1,H,-1)

        # === 局部邻域协方差（关键改进）===
        # 将 x 拆分为两组
        x1 = x[:, :C//2]   # [B, C/2, H, W]
        x2 = x[:, C//2:]   # [B, C/2, H, W]

        # 展开邻域（k×k×C/2）
        x1_unfold = self.unfold(x1)  # [B, (C/2)*k*k, H*W]
        x2_unfold = self.unfold(x2)  # [B, (C/2)*k*k, H*W]

        # 重塑为 [B, C/2, k*k, H*W]
        x1_unfold = x1_unfold.view(B, C//2, -1, H*W)
        x2_unfold = x2_unfold.view(B, C//2, -1, H*W)

        # 计算邻域内通道协方差（沿 k*k 维度）
        cov = (x1_unfold * x2_unfold).mean(dim=2)  # [B, C/2, H*W]
        consistency = cov.abs().mean(dim=1, keepdim=True)  # [B, 1, H*W]
        consistency = consistency.view(B, 1, H, W)

        # 异常分数：一致性越低，越异常
        local_anomaly = 1.0 - consistency.clamp(0, 1)

        # === 融合 ===
        fusion = global_dir.mean(dim=1, keepdim=True) * local_anomaly
        attention = torch.sigmoid(self.gate(fusion))

        return x * attention


class LDA(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.channels = channels
        mid_ch = max(8, channels // reduction)

        # 可学习方向权重（替代手工滤波器）
        self.dir_weight_h = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.dir_weight_w = nn.Parameter(torch.ones(1, channels, 1, 1))

        # 轻量门控
        self.gate = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, x):
        B, C, H, W = x.shape

        # === 1. 全局方向感知（可学习加权池化）===
        # 垂直方向敏感：沿 W 加权 → 响应水平结构
        weight_w = torch.softmax(self.dir_weight_w, dim=1)  # [1,C,1,1]
        global_h = (x * weight_w).mean(dim=3, keepdim=True)  # [B,C,H,1]

        # 水平方向敏感：沿 H 加权 → 响应垂直结构
        weight_h = torch.softmax(self.dir_weight_h, dim=1)
        global_w = (x * weight_h).mean(dim=2, keepdim=True)  # [B,C,1,W]

        # 融合全局方向上下文
        global_dir = global_h.expand(-1, -1, -1, W) + global_w.expand(-1, -1, H, -1)  # [B,C,H,W]

        # === 2. 局部方向异常（无滤波器！）===
        # 通过通道间协方差捕捉局部不一致性
        x_norm = x - x.mean(dim=1, keepdim=True)
        local_cov = (x_norm[:, :C//2] * x_norm[:, C//2:]).mean(dim=1, keepdim=True)  # [B,1,H,W]
        consistency = local_cov.abs()  # 高协方差 = 方向一致；低 = 异常
        local_anomaly = torch.relu(1.0 - consistency)

        # === 3. 融合 + 门控 ===
        fusion = global_dir.mean(dim=1, keepdim=True) * local_anomaly  # [B,1,H,W]
        attention = torch.sigmoid(self.gate(fusion))  # 或 h_sigmoid()

        return x * attention


class A2CA(nn.Module):
    def __init__(self, in_channels, reduction=32, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # 异常图生成（无参数！）
        self.mean_h = nn.Conv2d(in_channels, in_channels, (1, kernel_size),
                                padding=(0, padding), groups=in_channels, bias=False)
        self.mean_w = nn.Conv2d(in_channels, in_channels, (kernel_size, 1),
                                padding=(padding, 0), groups=in_channels, bias=False)
        nn.init.constant_(self.mean_h.weight, 1.0 / kernel_size)
        nn.init.constant_(self.mean_w.weight, 1.0 / kernel_size)

        # 标准 CA（你已优化的版本）
        self.ca = CoordinateAttention(in_channels, reduction=reduction)

    def forward(self, x):
        # Step 1: 生成方向异常图
        mu_h = self.mean_h(x)
        mu_w = self.mean_w(x)
        anomaly = (x - mu_h).abs() + (x - mu_w).abs()

        # Step 2: CA 作用于异常图
        attention = self.ca(anomaly)

        # Step 3: 用原始 x 加权（保留信息完整性）
        return x * attention


class LearnableCoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        mip = max(8, in_channels // reduction)
        self.kernel_size = kernel_size

        # 可学习局部池化（depth-wise conv）
        self.pool_h = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=in_channels,
            bias=False
        )
        self.pool_w = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=in_channels,
            bias=False
        )

        # 初始化为均值池化
        nn.init.constant_(self.pool_h.weight, 1.0 / kernel_size)
        nn.init.constant_(self.pool_w.weight, 1.0 / kernel_size)

        # 共享 MLP
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        identity = x

        # 可学习局部池化（保持空间尺寸）
        x_h = self.pool_h(x)  # [B, C, H, W]
        x_w = self.pool_w(x)  # [B, C, H, W]

        # 独立处理（简化版，避免 cat/split）
        y_h = self.act(self.bn(self.conv1(x_h)))
        y_w = self.act(self.bn(self.conv1(x_w)))

        a_h = self.conv_h(y_h).sigmoid()
        a_w = self.conv_w(y_w).sigmoid()

        return identity * a_h * a_w


class DLCA(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(1, in_channels // reduction)
        # 共享压缩层（节省参数）
        self.compress = nn.Conv2d(in_channels, mip, 1, bias=False)
        self.act = nn.Hardswish()  # 轻量激活

        # 独立扩展层（保持方向特异性）
        self.expand_h = nn.Conv2d(mip, in_channels, 1, bias=True)
        self.expand_w = nn.Conv2d(mip, in_channels, 1, bias=True)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)        # [B, C, H, 1]
        x_w = self.pool_w(x)        # [B, C, 1, W]

        # 共享压缩
        y_h = self.compress(x_h)    # [B, mip, H, 1]
        y_w = self.compress(x_w)    # [B, mip, 1, W]

        y_h = self.act(y_h)
        y_w = self.act(y_w)

        # 独立扩展
        a_h = self.expand_h(y_h).sigmoid()  # [B, C, H, 1]
        a_w = self.expand_w(y_w).sigmoid()  # [B, C, 1, W]

        out = identity * a_h * a_w
        return out


class EF_LDDA(nn.Module):
    def __init__(self, in_channels, kernel_size=5, scale_factor=8):
        super().__init__()
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        padding = kernel_size // 2

        # === 局部方向异常（LDDA，全分辨率）===
        self.mean_h = nn.Conv2d(in_channels, in_channels, (1, kernel_size),
                                padding=(0, padding), groups=in_channels, bias=False)
        self.mean_w = nn.Conv2d(in_channels, in_channels, (kernel_size, 1),
                                padding=(padding, 0), groups=in_channels, bias=False)
        nn.init.constant_(self.mean_h.weight, 1.0 / kernel_size)
        nn.init.constant_(self.mean_w.weight, 1.0 / kernel_size)
        self.attn_local = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels, bias=True)
        nn.init.constant_(self.attn_local.bias, -1.0)

        # === 全局对比异常：使用固定 stride 的 AvgPool 替代 adaptive pool ===
        # 注意：要求输入 H, W 能被 scale_factor 整除
        self.downsample_pool = nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x):
        identity = x

        # --- 局部方向注意力（全分辨率）---
        mu_h = self.mean_h(x)
        mu_w = self.mean_w(x)
        dev = (x - mu_h).abs() + (x - mu_w).abs()
        local_attn = self.attn_local(dev).sigmoid()

        # --- 全局对比注意力（低分辨率计算）---
        B, C, H, W = x.shape

        # 使用固定 kernel 的 AvgPool2d 降采样（ONNX 兼容）
        x_low = self.downsample_pool(x)  # shape: [B, C, H//s, W//s]

        # 计算低分辨率下的全局均值
        global_mean_low = x_low.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        global_dev_low = (x_low - global_mean_low).abs()       # [B, C, H//s, W//s]

        # 上采样回原分辨率（nearest 插值，ONNX 支持）
        global_attn = F.interpolate(global_dev_low, size=(H, W), mode='nearest')
        global_attn = torch.sigmoid(global_attn)

        # --- 融合 ---
        fused_attn = 0.7 * local_attn + 0.3 * global_attn

        return identity * fused_attn


class AEF_LDDA(nn.Module):
    def __init__(self, in_channels, kernel_size=5, scale_factor=8):
        super().__init__()
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        padding = kernel_size // 2

        # === 局部方向分支 ===
        self.mean_h = nn.Conv2d(in_channels, in_channels, (1, kernel_size),
                                padding=(0, padding), groups=in_channels, bias=False)
        self.mean_w = nn.Conv2d(in_channels, in_channels, (kernel_size, 1),
                                padding=(padding, 0), groups=in_channels, bias=False)
        nn.init.constant_(self.mean_h.weight, 1.0 / kernel_size)
        nn.init.constant_(self.mean_w.weight, 1.0 / kernel_size)
        self.attn_local = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels, bias=True)
        nn.init.constant_(self.attn_local.bias, -1.0)

        # === 轻量自适应门控（无参数！）===
        # 利用输入 x 的全局统计动态决定局部/全局权重
        # 不引入任何新参数，仅用现有张量计算

    def forward(self, x):
        identity = x

        # --- 局部注意力 ---
        mu_h = self.mean_h(x)
        mu_w = self.mean_w(x)
        dev = (x - mu_h).abs() + (x - mu_w).abs()
        local_attn = self.attn_local(dev).sigmoid()

        # --- 全局对比注意力（低分辨率）---
        B, C, H, W = x.shape
        x_low = F.adaptive_avg_pool2d(x, (H // self.scale_factor, W // self.scale_factor))
        global_mean_low = x_low.mean(dim=[2, 3], keepdim=True)
        global_dev_low = (x_low - global_mean_low).abs()
        global_attn = F.interpolate(global_dev_low, size=(H, W), mode='nearest')
        global_attn = torch.sigmoid(global_attn)

        # --- 自适应融合（无参数！）---
        # 用 x 的全局能量衡量“图像复杂度”
        # 若图像整体变化大（如高纹理），则信任局部；若整体平滑，则信任全局
        energy = x.abs().mean(dim=[1, 2, 3], keepdim=True)  # [B, 1, 1, 1]
        # 归一化到 [0.5, 0.9] 区间（经验）
        local_weight = 0.5 + 0.4 * torch.sigmoid(energy - 0.3)  # 可调偏置
        local_weight = local_weight.view(B, 1, 1, 1)

        fused_attn = local_weight * local_attn + (1 - local_weight) * global_attn

        return identity * fused_attn


class EfficientChannelCA(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.in_channels = in_channels
        mip = max(8, in_channels // reduction)

        # 保持原始池化方法
        self.pool_h = lambda x: torch.mean(x, dim=3, keepdim=True)
        self.pool_w = lambda x: torch.mean(x, dim=2, keepdim=True)

        # 共享卷积层 - 增加非线性
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mip, kernel_size=1),
            nn.ReLU(),  # 轻量激活
            nn.Conv2d(mip, mip, kernel_size=1),  # 增加深度但不增加通道
            nn.Hardswish()
        )

        # 分离的注意力头 - 增加偏置项增强表达能力
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, bias=True)

        # 轻量上下文增强
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(4, in_channels // 64), 1),  # 极小通道数
            nn.ReLU(),
            nn.Conv2d(max(4, in_channels // 64), in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape

        # 上下文增强
        context = self.context_gate(x)
        x = x * context

        # 水平方向编码
        x_h = self.pool_h(x)

        # 垂直方向编码
        x_w = self.pool_w(x)

        # 拼接并处理特征
        y = torch.cat([x_h, x_w.permute(0, 1, 3, 2)], dim=2)
        y = self.conv1(y)

        # 分离特征
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成注意力图
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


class LightCoordAtt(nn.Module):
    """
    轻量级坐标注意力模块
    核心优化：
    1. 可控下采样降低计算量
    2. 移除BN层提升部署友好性
    3. ReLU替代Hardswish加速推理
    4. 维度优化减少转置操作

    参数：
    - in_channels: 输入通道数
    - reduction: 压缩比例（默认32）
    - downsample: 下采样比例（默认4，设为1禁用）
    """

    def __init__(self, in_channels, reduction=32, downsample=4):
        super().__init__()
        self.downsample = downsample
        self.in_channels = in_channels
        mip = max(8, in_channels // reduction)  # 确保最小通道数

        # 下采样层（可选）
        if downsample > 1:
            self.downsample_layer = nn.AvgPool2d(kernel_size=downsample)
            self.upsample_layer = nn.Upsample(
                scale_factor=downsample,
                mode='bilinear',
                align_corners=False
            )
        else:
            self.downsample_layer = nn.Identity()
            self.upsample_layer = nn.Identity()

        # 特征变换层
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.act = nn.ReLU(inplace=True)  # 轻量激活函数

        # 分支注意力生成
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 卷积层初始化
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.conv_h.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.conv_w.weight, mode='fan_out')

        # 偏置初始化
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv_h.bias, 0)
        nn.init.constant_(self.conv_w.bias, 0)

    def forward(self, x):
        identity = x

        # 下采样处理
        x_ds = self.downsample_layer(x)
        n, c, h_ds, w_ds = x_ds.shape

        # 水平方向编码 [n, c, h, 1]
        x_h = x_ds.mean(dim=3, keepdim=True)  # shape: [n, c, h_ds, 1]

        # 垂直方向编码 [n, c, 1, w]
        x_w = x_ds.mean(dim=2, keepdim=True)  # shape: [n, c, 1, w_ds]

        # 解决维度不匹配问题:
        # 1. 转置x_w以匹配维度
        x_w = x_w.permute(0, 1, 3, 2)  # 现在shape: [n, c, w_ds, 1]

        # 2. 通道融合（在dim=2上拼接）
        fused = torch.cat([x_h, x_w], dim=2)  # shape: [n, c, h_ds + w_ds, 1]

        # 特征变换
        fused = self.conv1(fused)
        fused = self.act(fused)

        # 分离特征
        split_h = fused[:, :, :h_ds, :]  # shape: [n, c, h_ds, 1]
        split_w = fused[:, :, h_ds:, :]  # shape: [n, c, w_ds, 1]

        # 恢复x_w的原始维度顺序
        split_w = split_w.permute(0, 1, 3, 2)  # shape: [n, c, 1, w_ds]

        # 注意力图生成
        att_h = self.conv_h(split_h).sigmoid()  # shape: [n, c, h_ds, 1]
        att_w = self.conv_w(split_w).sigmoid()  # shape: [n, c, 1, w_ds]

        # 上采样恢复原始分辨率
        # 扩展高度注意力到完整宽度
        att_h = att_h.expand(-1, -1, -1, w_ds)  # shape: [n, c, h_ds, w_ds]
        # 扩展宽度注意力到完整高度
        att_w = att_w.expand(-1, -1, h_ds, -1)  # shape: [n, c, h_ds, w_ds]

        # 上采样到原始分辨率
        att_h = self.upsample_layer(att_h)
        att_w = self.upsample_layer(att_w)

        # 应用注意力（残差连接）
        return identity * att_h * att_w


if __name__ == '__main__':
    x = torch.rand(1, 64, 80, 80)
    # up = F.interpolate(x, scale_factor=2., mode='nearest')
    dys = CoordinateAttention(64)
    edys = EfficientChannelCA(64)
    from Addmodule.cal import calculate_module_stats

    # calculate_module_stats(up, x)
    calculate_module_stats(dys, x)
    calculate_module_stats(edys, x)