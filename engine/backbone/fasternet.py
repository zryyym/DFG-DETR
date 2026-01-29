# engine/backbone/fasternet.py
from functools import partial

import torch
import torch.nn as nn
from .common import FrozenBatchNorm2d
from ..core import register
import os
import logging
from timm.models.layers import DropPath

__all__ = ['FasterNet']


# ==================== DropPath (from timm) ====================
# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         if self.drop_prob == 0. or not self.training:
#             return x
#         keep_prob = 1 - self.drop_prob
#         shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#         random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
#         if keep_prob > 0:
#             random_tensor.div_(keep_prob)
#         return x * random_tensor


# ==================== Partial_conv3 (exact copy from official) ====================
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


# ==================== MLPBlock (exact as official) ====================
class MLPBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


# ==================== BasicStage (official style) ====================
class BasicStage(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):
        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        return self.blocks(x)


# ==================== PatchEmbed (official) ====================
class PatchEmbed(nn.Module):
    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


# ==================== PatchMerging (official) ====================
class PatchMerging(nn.Module):
    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.reduction(x))


# ==================== FasterNetBackbone (adapted for detection) ====================
@register()
class FasterNet(nn.Module):
    """
    FasterNet backbone adapted from official implementation.
    No structural modification — strictly follows https://github.com/JierunChen/FasterNet

    Args:
        name (str): 's', 'm', 'l'
        return_idx (list[int]): stage indices to return (e.g., [1,2,3] for strides 8,16,32)
        freeze_at (int): freeze stem and first `freeze_at` stages
        freeze_norm (bool): replace BN with FrozenBatchNorm2d
        pretrained (bool): load ImageNet pretrained weights
        local_model_dir (str): directory to save/load pretrained models
    """

    arch_settings = {
        't0': dict(embed_dim=40, depths=[1, 2, 8, 2], drop_path_rate=0.0),
        't1': dict(embed_dim=64, depths=[1, 2, 8, 2], drop_path_rate=0.02),
        't2': dict(embed_dim=96, depths=[1, 2, 8, 2], drop_path_rate=0.05),
        's': dict(embed_dim=128, depths=[1, 2, 13, 2], drop_path_rate=0.1),
        'm': dict(embed_dim=144, depths=[3, 4, 18, 3], drop_path_rate=0.2),
        'l': dict(embed_dim=192, depths=[3, 4, 18, 3], drop_path_rate=0.3),
    }

    model_urls = {
        't0': 'https://github.com/JierunChen/FasterNet/releases/download/v1.0/fn_t0.pth',
        't1': 'https://github.com/JierunChen/FasterNet/releases/download/v1.0/fn_t1.pth',
        't2': 'https://github.com/JierunChen/FasterNet/releases/download/v1.0/fn_t2.pth',
        's': 'https://github.com/JierunChen/FasterNet/releases/download/v1.0/fn_s.pth',
        'm': 'https://github.com/JierunChen/FasterNet/releases/download/v1.0/fn_m.pth',
        'l': 'https://github.com/JierunChen/FasterNet/releases/download/v1.0/fn_l.pth',
    }

    def __init__(
        self,
        name='s',
        return_idx=[1, 2, 3],
        freeze_at=0,
        freeze_norm=True,
        pretrained=False,
        local_model_dir='weight/fasternet/',
        n_div=4,
        mlp_ratio=2.0,
        layer_scale_init_value=0.0,
        pconv_fw_type='split_cat',
        norm_layer='BN',
        act_layer='RELU',
    ):
        super().__init__()
        assert name in self.arch_settings, f"Unsupported variant: {name}"
        self.return_idx = return_idx

        cfg = self.arch_settings[name]
        embed_dim = cfg['embed_dim']
        depths = cfg['depths']
        drop_path_rate = cfg['drop_path_rate']

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        # === Stem ===
        self.patch_embed = PatchEmbed(
            patch_size=4,
            patch_stride=4,
            in_chans=3,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        # === Build stages (interleaved BasicStage + PatchMerging) ===
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        stages_and_merges = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            # BasicStage
            stage = BasicStage(
                dim=int(embed_dim * (2 ** i)),
                depth=depths[i],
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[cur:cur + depths[i]],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            stages_and_merges.append(stage)
            cur += depths[i]

            # PatchMerging (except last stage)
            if i < len(depths) - 1:
                merge = PatchMerging(
                    patch_size2=2,
                    patch_stride2=2,
                    dim=int(embed_dim * 2 ** i),
                    norm_layer=norm_layer
                )
                stages_and_merges.append(merge)

        self.stages = stages_and_merges
        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [
            embed_dim,
            embed_dim * 2,
            embed_dim * 4,
            embed_dim * 8
        ]

        # Freeze parameters
        if freeze_at >= 0:
            self._freeze_parameters(self.patch_embed)
            for i in range(min(freeze_at * 2, len(self.stages))):  # each stage+merge counts as 2
                self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            self._load_pretrained(name, local_model_dir)

    def _freeze_parameters(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def _freeze_norm(self, module):
        if isinstance(module, nn.BatchNorm2d):
            return FrozenBatchNorm2d(module.num_features)
        for name, child in module.named_children():
            new_child = self._freeze_norm(child)
            if new_child is not child:
                setattr(module, name, new_child)
        return module

    def _load_pretrained(self, name, local_dir):
        RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
        url = self.model_urls[name]
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, f'fn_{name}.pth')
        try:
            if os.path.exists(local_path):
                state_dict = torch.load(local_path, map_location='cpu')
            else:
                state_dict = torch.hub.load_state_dict_from_url(url, model_dir=local_dir, map_location='cpu')

            # Remove classifier head keys
            clean_state = {}
            for k, v in state_dict.items():
                if not k.startswith(('head', 'avgpool_pre_head')):
                    clean_state[k] = v

            missing, unexpected = self.load_state_dict(clean_state, strict=False)
            print(GREEN + f"FasterNet-{name.upper()} pretrained loaded." + RESET)
            if missing:
                logging.warning(f"Missing keys: {missing}")
            if unexpected:
                logging.warning(f"Unexpected keys: {unexpected}")
        except Exception as e:
            logging.error(RED + f"Failed to load FasterNet-{name}: {e}" + RESET)

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        stage_idx = 0  # logical stage index (0～3)
        for i, layer in enumerate(self.stages):
            x = layer(x)
            # After BasicStage (even indices: 0,2,4,6), we have a complete stage output
            if i % 2 == 0:
                if stage_idx in self.return_idx:
                    outs.append(x)
                stage_idx += 1
        return outs