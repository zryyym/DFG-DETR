import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core import register
from .common import FrozenBatchNorm2d
import os
import logging

__all__ = ['MobileNetV3']


# # ==================== 1. Block: Inverted Residual with SE & Hardswish ====================
# def _make_divisible(v, divisor=8, min_value=None):
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v


class Hardsigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6


class Hardswish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


class SqueezeExcitation(nn.Module):
    def __init__(self, in_chs, reduction=4):
        super().__init__()
        reduced_chs = max(in_chs // 4, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chs, reduced_chs, 1, bias=False),
            nn.BatchNorm2d(reduced_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_chs, in_chs, 1, bias=False),
            Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidual(nn.Module):
    """MobileNetV3 Basic Block"""

    def __init__(self, in_chs, out_chs, kernel_size, stride, exp_chs, use_se=False, act_layer=nn.ReLU):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = (stride == 1 and in_chs == out_chs)

        layers = []
        # Expand
        if exp_chs != in_chs:
            layers.extend([
                nn.Conv2d(in_chs, exp_chs, 1, bias=False),
                nn.BatchNorm2d(exp_chs),
                act_layer(inplace=True)
            ])
        # Depthwise
        layers.extend([
            nn.Conv2d(exp_chs, exp_chs, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=exp_chs, bias=False),
            nn.BatchNorm2d(exp_chs),
            act_layer(inplace=True)
        ])
        # SE
        if use_se:
            layers.append(SqueezeExcitation(exp_chs))
        # Project
        layers.extend([
            nn.Conv2d(exp_chs, out_chs, 1, bias=False),
            nn.BatchNorm2d(out_chs)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


# ==================== 2. Stage: Group of Blocks + Optional Downsample ====================
class MBV3_Stage(nn.Module):
    """
    A stage in MobileNetV3, consisting of multiple InvertedResidual blocks.
    The first block may perform downsampling (stride=2).
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            block_configs,  # List of [kernel, exp, use_se, act, stride]
            use_downsample=True,
            downsample_mode='default',  # 'default' or custom (e.g., 'drfd', 'wt', etc.)
            **kwargs
    ):
        super().__init__()
        self.blocks = nn.Sequential()
        current_chs = in_chs

        for i, (k, exp, se, act, s) in enumerate(block_configs):
            act_layer = nn.Hardswish if act == 'hardswish' else nn.ReLU
            # Only the first block can downsample
            stride = s if i == 0 else 1
            block = InvertedResidual(
                current_chs, out_chs, k, stride, exp, se, act_layer
            )
            self.blocks.add_module(f"block{i}", block)
            current_chs = out_chs

        # Optional: replace default downsample with custom module (future extension)
        self.downsample_mode = downsample_mode

    def forward(self, x):
        return self.blocks(x)


# ==================== 3. Backbone: Stem + Stages + Output Control ====================
@register()
class MobileNetV3(nn.Module):
    """
    MobileNetV3 Backbone with clear block-stage structure.
    Compatible with your framework: return_idx, freeze, pretrained, etc.
    """
    arch_settings = {
        'small': {
            'stem_chs': 16,
            'final_chs': 1024,
            'stages': [
                # Each stage: [out_chs, [[k, exp, se, act, stride], ...]]
                [16, [[3, 16, True, 'relu', 2]]],  # stage0 → stride=4
                [24, [[3, 72, False, 'relu', 2],
                      [3, 88, False, 'relu', 1]]],  # stage1 → stride=8
                [48, [[5, 96, True, 'hardswish', 2],
                      [5, 240, True, 'hardswish', 1],
                      [5, 240, True, 'hardswish', 1],
                      [5, 120, True, 'hardswish', 1],
                      [5, 144, True, 'hardswish', 1]]],  # stage2 → stride=16
                [96, [[5, 288, True, 'hardswish', 2],
                      [5, 576, True, 'hardswish', 1],
                      [5, 576, True, 'hardswish', 1]]],  # stage3 → stride=32
            ],
            'url': 'https://github.com/d-li14/mobilenetv3.pytorch/raw/master/pretrained/mobilenetv3-small-55df8e1f.pth'
        },
        'large': {
            'stem_chs': 16,
            'final_chs': 1280,
            'stages': [
                [16, [[3, 16, False, 'relu', 1]]],
                [24, [[3, 64, False, 'relu', 2],
                      [3, 72, False, 'relu', 1]]],
                [40, [[5, 72, True, 'relu', 2],
                      [5, 120, True, 'relu', 1],
                      [5, 120, True, 'relu', 1]]],
                [112, [[3, 240, False, 'hardswish', 2],
                      [3, 200, False, 'hardswish', 1],
                      [3, 184, False, 'hardswish', 1],
                      [3, 184, False, 'hardswish', 1],
                      [3, 480, True, 'hardswish', 1],
                      [3, 672, True, 'hardswish', 1]]],
                [160, [[5, 672, True, 'hardswish', 2],
                       [5, 960, True, 'hardswish', 1],
                       [5, 960, True, 'hardswish', 1]]],
            ],
            'url': 'https://github.com/d-li14/mobilenetv3.pytorch/raw/master/pretrained/mobilenetv3-large-1cd25616.pth'
        }
    }

    def __init__(
            self,
            name='small',
            return_idx=[1, 2, 3],  # stage indices to return (0-based)
            freeze_at=0,
            freeze_norm=True,
            pretrained=False,
            local_model_dir='weight/mobilenetv3/',
            downsample_mode='default',  # reserved for future custom downsample
    ):
        super().__init__()
        assert name in ['small', 'large']
        config = self.arch_settings[name]
        self.return_idx = return_idx

        # === Stem ===
        stem_act = Hardswish if name == 'large' else nn.ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(3, config['stem_chs'], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config['stem_chs']),
            nn.Hardswish(),
        )

        # === Stages ===
        in_chs = config['stem_chs']
        self.stages = nn.ModuleList()
        for out_chs, block_cfgs in config['stages']:
            stage = MBV3_Stage(
                in_chs=in_chs,
                out_chs=out_chs,
                block_configs=block_cfgs,
                downsample_mode=downsample_mode
            )
            self.stages.append(stage)
            in_chs = out_chs

        # === Output Info ===
        self._out_strides = [4, 8, 16, 32][:len(self.stages)]
        self._out_channels = [stage.blocks[-1].block[-2].out_channels for stage in self.stages]

        # === Freeze ===
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            for i in range(min(freeze_at, len(self.stages))):
                self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        # === Pretrained ===
        if pretrained:
            self._load_pretrained(name, local_model_dir, config['url'])

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

    def _load_pretrained(self, name, local_dir, url):
        RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, f'mobilenetv3-{name}.pth')
        try:
            if os.path.exists(local_path):
                state_dict = torch.load(local_path, map_location='cpu')
            else:
                state_dict = torch.hub.load_state_dict_from_url(url, model_dir=local_dir, map_location='cpu')

            # Map keys: remove 'classifier', adapt 'features'
            new_state = {}
            for k, v in state_dict.items():
                if k.startswith('features'):
                    new_k = k.replace('features.', '')
                    new_state[new_k] = v
                elif k in ['conv_stem.weight', 'bn1.weight']:  # adjust if needed
                    new_state[k] = v
            self.load_state_dict(new_state, strict=False)
            print(GREEN + f"Pretrained MobileNetV3-{name} loaded." + RESET)
        except Exception as e:
            logging.error(RED + f"Failed to load pretrained: {e}" + RESET)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs