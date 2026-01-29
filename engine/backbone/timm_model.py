"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#0583
"""
import timm
import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from engine.backbone.utils import IntermediateLayerGetter
from torchvision.models.feature_extraction import create_feature_extractor
from engine.core import register

from timm import models

# @register()
class TimmModel(torch.nn.Module):
    def __init__(
        self,
        name,
        return_layers,  # e.g., {'blocks.3': 'feat1', 'blocks.5': 'feat2'}
        pretrained=False,
        exportable=True,
        features_only=False,  # ⚠️ 关键：设为 False！
        **kwargs
    ):
        super().__init__()

        # 必须关闭 features_only，否则模型是 FeatureListNet，无法用 create_feature_extractor
        model = timm.create_model(
            name,
            pretrained=pretrained,
            exportable=exportable,
            features_only=False,  # ←←← 重要！
            **kwargs
        )

        # 验证所有 return_layers keys 是否存在
        all_names = {name for name, _ in model.named_modules()}
        missing = set(return_layers) - all_names
        if missing:
            raise ValueError(f"Modules not found in model: {missing}. "
                             f"Example available: {[n for n in all_names if 'block' in n or 'stage' in n][:5]}")

        # 使用 torchvision 的 create_feature_extractor
        self.model = create_feature_extractor(model, return_nodes=return_layers)

        # 注意：此时无法自动获取 strides/channels（除非手动计算或查表）
        self.strides = None
        self.channels = None
        self.return_layers = return_layers

    def forward(self, x):
        out = self.model(x)  # OrderedDict
        return list(out.values())  # 保持和原来一致的输出格式


if __name__ == '__main__':

    model = TimmModel(name='mobilenetv3_small_100', return_layers=['blocks.3', 'blocks.5'])
    data = torch.rand(1, 3, 640, 640)
    outputs = model(data)

    for output in outputs:
        print(output.shape)

    """
    model:
        type: TimmModel
        name: resnet34
        return_layers: ['layer2', 'layer4']
    """
