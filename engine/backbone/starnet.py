# engine/backbone/starnet.py

import torch
import torch.nn as nn
from .common import FrozenBatchNorm2d
from ..core import register
import os
import logging

__all__ = ['StarNet']


# ==================== 1. Block (from official StarNet) ====================
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class StarBlock(nn.Module):
    """StarNet Block with element-wise multiplication"""

    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


# ==================== 2. DropPath (copy from timm) ====================
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# ==================== 3. Stage ====================
class StarStage(nn.Module):
    def __init__(self, in_chs, embed_dim, block_num, mlp_ratio=4, drop_path_rates=None):
        super().__init__()
        # Downsample: ordinary conv (not depthwise!) to change channel and downsample
        self.downsample = ConvBN(in_chs, embed_dim, kernel_size=3, stride=2, padding=1, with_bn=True)
        # Blocks
        blocks = []
        dprs = drop_path_rates or [0.0] * block_num
        for i in range(block_num):
            blocks.append(StarBlock(embed_dim, mlp_ratio=mlp_ratio, drop_path=dprs[i]))
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = embed_dim

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


# ==================== 4. Backbone ====================
@register()
class StarNet(nn.Module):
    """
    StarNet Backbone for detection/segmentation.
    Reference: "Rewrite the Stars" (https://arxiv.org/abs/2403.19967)

    Args:
        name (str): starnet_s1 ～ s4, or s050/s100/s150
        return_idx (list): indices of stages to return (default: [1,2,3] → stride 8,16,32)
        freeze_at (int): freeze stem and first `freeze_at` stages
        freeze_norm (bool): use FrozenBatchNorm2d
        pretrained (bool): load pretrained weights
        local_model_dir (str): local dir for pretrained models
    """

    arch_settings = {
        's050': dict(base_dim=16, depths=[1, 1, 3, 1], mlp_ratio=3),
        's100': dict(base_dim=20, depths=[1, 2, 4, 1], mlp_ratio=4),
        's150': dict(base_dim=24, depths=[1, 2, 4, 2], mlp_ratio=3),
        's1': dict(base_dim=24, depths=[2, 2, 8, 3], mlp_ratio=4),
        's2': dict(base_dim=32, depths=[1, 2, 6, 2], mlp_ratio=4),
        's3': dict(base_dim=32, depths=[2, 2, 8, 4], mlp_ratio=4),
        's4': dict(base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4),
    }

    model_urls = {
        "s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
        "s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
        "s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
        "s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
    }

    def __init__(
            self,
            name='s1',
            return_idx=[1, 2, 3],
            freeze_at=0,
            freeze_norm=True,
            pretrained=False,
            local_model_dir='weight/starnet/',
    ):
        super().__init__()
        assert name in self.arch_settings, f"Unsupported StarNet variant: {name}"
        self.return_idx = return_idx

        cfg = self.arch_settings[name]
        base_dim = cfg['base_dim']
        depths = cfg['depths']
        mlp_ratio = cfg['mlp_ratio']

        # === Stem ===
        self.stem = nn.Sequential(
            ConvBN(3, 32, kernel_size=3, stride=2, padding=1, with_bn=True),
            nn.ReLU6()
        )
        current_chs = 32

        # === Stages ===
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule
        self.stages = nn.ModuleList()
        cur = 0
        for i, depth in enumerate(depths):
            embed_dim = base_dim * (2 ** i)
            stage_dprs = dpr[cur:cur + depth]
            stage = StarStage(current_chs, embed_dim, depth, mlp_ratio, stage_dprs)
            self.stages.append(stage)
            current_chs = embed_dim
            cur += depth

        # Output info
        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage.out_channels for stage in self.stages]

        # Freeze
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            for i in range(min(freeze_at, len(self.stages))):
                self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained and name in self.model_urls:
            self._load_pretrained(name, local_model_dir)

    def _freeze_parameters(self, m):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m):
        if isinstance(m, nn.BatchNorm2d):
            return FrozenBatchNorm2d(m.num_features)
        for name, child in m.named_children():
            new_child = self._freeze_norm(child)
            if new_child is not child:
                setattr(m, name, new_child)
        return m

    def _load_pretrained(self, name, local_dir):
        RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
        url = self.model_urls[name]
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, f'starnet_{name}.pth.tar')
        try:
            if os.path.exists(local_path):
                checkpoint = torch.load(local_path, map_location='cpu')
            else:
                checkpoint = torch.hub.load_state_dict_from_url(url, model_dir=local_dir, map_location='cpu')

            state_dict = checkpoint["state_dict"]
            # Remove classifier-related keys
            new_state = {}
            for k, v in state_dict.items():
                if not k.startswith(('head', 'norm', 'avgpool')):
                    # Adapt key: e.g., 'stages.0.downsample.conv' → keep as is
                    new_state[k] = v
            self.load_state_dict(new_state, strict=False)
            print(GREEN + f"Pretrained StarNet-{name} loaded." + RESET)
        except Exception as e:
            logging.error(RED + f"Failed to load pretrained StarNet-{name}: {e}" + RESET)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs