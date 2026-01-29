# models/backbones/fastvit.py

import os
import logging
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from .common import FrozenBatchNorm2d  # 假设你有这个
from ..core import register  # 注册机制


__all__ = ['FastViT']


class SEBlock(nn.Module):
    """Squeeze and Excite module.

    Pytorch implementation of `Squeeze-and-Excitation Networks` -
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """Construct a Squeeze and Excite Module.

        Args:
            in_channels: Number of input channels.
            rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """MobileOne building block.

    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        """Construct a MobileOneBlock module.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the block.
            kernel_size: Size of the convolution kernel.
            stride: Stride size.
            padding: Zero-padding size.
            dilation: Kernel dilation factor.
            groups: Group number.
            inference_mode: If True, instantiates model in inference mode.
            use_se: Whether to use SE-ReLU activations.
            use_act: Whether to use activation. Default: ``True``
            use_scale_branch: Whether to use scale branch. Default: ``True``
            num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        if use_act:
            self.activation = activation
        else:
            self.activation = nn.Identity()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            # Re-parameterizable conv branches
            if num_conv_branches > 0:
                rbr_conv = list()
                for _ in range(self.num_conv_branches):
                    rbr_conv.append(
                        self._conv_bn(kernel_size=kernel_size, padding=padding)
                    )
                self.rbr_conv = nn.ModuleList(rbr_conv)
            else:
                self.rbr_conv = None

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if (kernel_size > 1) and use_scale_branch:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(
        self, branch: Union[nn.Sequential, nn.BatchNorm2d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Args:
            branch: Sequence of ops to be fused.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        """Helper method to construct conv-batchnorm layers.

        Args:
            kernel_size: Size of the convolution kernel.
            padding: Zero-padding size.

        Returns:
            Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class ReparamLargeKernelConv(nn.Module):
    """Building Block of RepLKNet

    This class defines overparameterized large kernel conv block
    introduced in `RepLKNet <https://arxiv.org/abs/2203.06717>`_

    Reference: https://github.com/DingXiaoH/RepLKNet-pytorch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int,
        inference_mode: bool = False,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        """Construct a ReparamLargeKernelConv module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of the large kernel conv branch.
            stride: Stride size. Default: 1
            groups: Group number. Default: 1
            small_kernel: Kernel size of small kernel conv branch.
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
            activation: Activation module. Default: ``nn.GELU``
        """
        super(ReparamLargeKernelConv, self).__init__()

        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.padding = kernel_size // 2
        if inference_mode:
            self.lkb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = self._conv_bn(
                kernel_size=kernel_size, padding=self.padding
            )
            if small_kernel is not None:
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = self._conv_bn(
                    kernel_size=small_kernel, padding=small_kernel // 2
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, "small_conv"):
                out += self.small_conv(x)

        self.activation(out)
        return out

    def get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepLKNet-pytorch

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        eq_k, eq_b = self._fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def reparameterize(self) -> None:
        """
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        eq_k, eq_b = self.get_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.groups,
            bias=True,
        )

        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

    @staticmethod
    def _fuse_bn(
        conv: torch.Tensor, bn: nn.BatchNorm2d
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with conv layer.

        Args:
            conv: Convolutional kernel weights.
            bn: Batchnorm 2d layer.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int = 0) -> nn.Sequential:
        """Helper method to construct conv-batchnorm layers.

        Args:
            kernel_size: Size of the convolution kernel.
            padding: Zero-padding size.

        Returns:
            A nn.Sequential Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class RepMixer(nn.Module):
    """Reparameterizable token mixer.

    For more details, please refer to our paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
    ):
        """Build RepMixer Module.

        Args:
            dim: Input feature map dimension. :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, H, W)`.
            kernel_size: Kernel size for spatial mixing. Default: 3
            use_layer_scale: If True, learnable layer scale is used. Default: ``True``
            layer_scale_init_value: Initial value for layer scale. Default: 1e-5
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x

    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = (
                self.mixer.id_tensor
                + self.mixer.reparam_conv.weight
                - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for para in self.parameters():
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")


class ConvFFN(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, padding=3, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RepMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, mlp_ratio=4.0, act_layer=nn.GELU, drop=0.0, drop_path=0.0,
                 use_layer_scale=True, layer_scale_init_value=1e-5, inference_mode=False):
        super().__init__()
        self.token_mixer = RepMixer(dim, kernel_size, use_layer_scale, layer_scale_init_value, inference_mode)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)))

    def forward(self, x):
        x = self.token_mixer(x)
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = x + self.drop_path(self.convffn(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, stride, in_channels, embed_dim, inference_mode=False):
        super().__init__()
        self.proj = nn.Sequential(
            ReparamLargeKernelConv(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
                small_kernel=3,
                inference_mode=inference_mode,
            ),
            MobileOneBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,  # ← 关键修复！
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            )
        )

    def forward(self, x):
        return self.proj(x)

# ==================== FastViT Backbone for Detection/Segmentation ====================
@register()
class FastViT(nn.Module):
    """
    FastViT backbone adapted from official implementation.
    Supports multi-scale feature output for dense prediction.

    Args:
        name (str): 't8', 't12', 's12', 'sa12', 'sa24', 'sa36', 'ma36'
        return_idx (list[int]): stage indices to return (e.g., [1,2,3])
        freeze_at (int): freeze stem and first `freeze_at` stages
        freeze_norm (bool): replace BN with FrozenBatchNorm2d
        pretrained (bool): load ImageNet pretrained weights
        local_model_dir (str): directory to save/load pretrained models
        inference_mode (bool): enable reparameterized inference (default: False during training)
    """

    arch_settings = {
        't8':  dict(layers=[2,2,4,2],   embed_dims=[48,96,192,384],   mlp_ratios=[3,3,3,3],   token_mixers=('repmixer',)*4),
        't12': dict(layers=[2,2,6,2],   embed_dims=[64,128,256,512],  mlp_ratios=[3,3,3,3],   token_mixers=('repmixer',)*4),
        's12': dict(layers=[2,2,6,2],   embed_dims=[64,128,256,512],  mlp_ratios=[4,4,4,4],   token_mixers=('repmixer',)*4),
        'sa12':dict(layers=[2,2,6,2],   embed_dims=[64,128,256,512],  mlp_ratios=[4,4,4,4],   token_mixers=('repmixer','repmixer','repmixer','attention')),
        'sa24':dict(layers=[4,4,12,4],  embed_dims=[64,128,256,512],  mlp_ratios=[4,4,4,4],   token_mixers=('repmixer','repmixer','repmixer','attention')),
        'sa36':dict(layers=[6,6,18,6],  embed_dims=[64,128,256,512],  mlp_ratios=[4,4,4,4],   token_mixers=('repmixer','repmixer','repmixer','attention')),
        'ma36':dict(layers=[6,6,18,6],  embed_dims=[76,152,304,608],  mlp_ratios=[4,4,4,4],   token_mixers=('repmixer','repmixer','repmixer','attention')),
    }

    model_urls = {
        't8':  'https://docs-assets.developer.apple.com/ml-research/models/fastvit/fastvit_t8.pt',
        't12': 'https://docs-assets.developer.apple.com/ml-research/models/fastvit/fastvit_t12.pt',
        's12': 'https://docs-assets.developer.apple.com/ml-research/models/fastvit/fastvit_s12.pt',
        'sa12':'https://docs-assets.developer.apple.com/ml-research/models/fastvit/fastvit_sa12.pt',
        'sa24':'https://docs-assets.developer.apple.com/ml-research/models/fastvit/fastvit_sa24.pt',
        'sa36':'https://docs-assets.developer.apple.com/ml-research/models/fastvit/fastvit_sa36.pt',
        'ma36':'https://docs-assets.developer.apple.com/ml-research/models/fastvit/fastvit_ma36.pt',
    }

    def __init__(
        self,
        name='s12',
        return_idx=[1, 2, 3],
        freeze_at=0,
        freeze_norm=True,
        pretrained=False,
        local_model_dir='weight/fastvit/',
        inference_mode=False,
        **kwargs
    ):
        super().__init__()
        assert name in self.arch_settings, f"Unsupported variant: {name}"
        self.return_idx = return_idx
        self.inference_mode = inference_mode

        cfg = self.arch_settings[name]
        layers = cfg['layers']
        embed_dims = cfg['embed_dims']
        mlp_ratios = cfg['mlp_ratios']
        token_mixers = cfg['token_mixers']

        # === Stem ===
        self.stem = nn.Sequential(
            MobileOneBlock(
                in_channels=3,
                out_channels=embed_dims[0],
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            ),
            MobileOneBlock(
                in_channels=embed_dims[0],
                out_channels=embed_dims[0],
                kernel_size=3,
                stride=2,
                padding=1,
                groups=embed_dims[0],  # depthwise
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            ),
            MobileOneBlock(
                in_channels=embed_dims[0],
                out_channels=embed_dims[0],
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            ),
        )

        # === Build stages + downsampling ===
        self.stages = nn.ModuleList()
        self._out_strides = [4, 8, 16, 32]
        self._out_channels = embed_dims

        # Compute drop path rate
        total_blocks = sum(layers)
        dpr = [x.item() for x in torch.linspace(0, 0.1, total_blocks)]  # default drop_path_rate=0.1

        cur = 0
        for i in range(4):
            # Stage blocks
            stage_blocks = []
            for j in range(layers[i]):
                block_dpr = dpr[cur + j]
                if token_mixers[i] == 'repmixer':
                    block = RepMixerBlock(
                        dim=embed_dims[i],
                        kernel_size=3,
                        mlp_ratio=mlp_ratios[i],
                        drop=0.0,
                        drop_path=block_dpr,
                        use_layer_scale=True,
                        layer_scale_init_value=1e-5,
                        inference_mode=inference_mode
                    )
                else:  # attention
                    raise NotImplementedError("MHSA not implemented in this simplified version")
                stage_blocks.append(block)
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += layers[i]

            # Downsampling (except last stage)
            if i < 3:
                downsample = PatchEmbed(
                    patch_size=7,
                    stride=2,
                    in_channels=embed_dims[i],
                    embed_dim=embed_dims[i+1],
                    inference_mode=inference_mode
                )
                self.stages.append(downsample)

        # Freeze
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            for i in range(min(freeze_at * 2, len(self.stages))):
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
        url = self.model_urls.get(name)
        if url is None:
            logging.warning(f"No URL for {name}")
            return

        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, f'fastvit_{name}.pt')
        try:
            if os.path.exists(local_path):
                checkpoint = torch.load(local_path, map_location='cpu')
            else:
                checkpoint = torch.hub.load_state_dict_from_url(url, model_dir=local_dir, map_location='cpu')

            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

            # Remove classifier head
            clean_state = {}
            for k, v in state_dict.items():
                if not k.startswith(('head', 'gap', 'conv_exp')):
                    # Map keys: e.g., network.0 -> stem; network.1 -> stages.0; network.2 -> stages.1, etc.
                    new_k = k
                    if k.startswith('patch_embed'):
                        new_k = k.replace('patch_embed', 'stem')
                    elif k.startswith('network'):
                        idx = int(k.split('.')[1])
                        if idx == 0:
                            continue  # skip pos_emb if any
                        elif idx % 2 == 1:
                            stage_id = (idx - 1) // 2
                            new_k = k.replace(f'network.{idx}', f'stages.{stage_id}')
                        else:
                            merge_id = (idx - 2) // 2
                            new_k = k.replace(f'network.{idx}', f'stages.{merge_id + 1}')
                    clean_state[new_k] = v

            missing, unexpected = self.load_state_dict(clean_state, strict=False)
            print(GREEN + f"FastViT-{name.upper()} pretrained loaded." + RESET)
            if missing:
                logging.warning(f"Missing keys: {missing}")
            if unexpected:
                logging.warning(f"Unexpected keys: {unexpected}")

        except Exception as e:
            logging.error(RED + f"Failed to load FastViT-{name}: {e}" + RESET)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        stage_idx = 0
        for i, layer in enumerate(self.stages):
            x = layer(x)
            if i % 2 == 0:  # after stage block (not after downsample)
                if stage_idx in self.return_idx:
                    outs.append(x)
                stage_idx += 1
        return outs