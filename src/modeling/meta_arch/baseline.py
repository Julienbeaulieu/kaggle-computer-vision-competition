import torch
from torch import nn
from yacs.config import CfgNode
from src.modeling.backbone.build import build_backbone
from src.modeling.prediction_head.build import build_head
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register('baseline')
def build_baseline_model(model_cfg: CfgNode) -> nn.Module:
    return BaselineModel(model_cfg)


class BaselineModel(nn.Module):

    def __init__(self, model_cfg: CfgNode):
        super(BaselineModel, self).__init__()
        self.backbone = build_backbone(model_cfg.BACKBONE)
        self.head = build_head(model_cfg.HEAD)
        self.heads_dims = model_cfg.HEAD.OUTPUT_DIMS

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        grapheme_logits, vowel_logits, consonant_logits = torch.split(x, self.heads_dims, dim=1)
        return grapheme_logits, vowel_logits, consonant_logits

    def freeze_bn(self):
        """
        https://github.com/kuangliu/pytorch-retinanet/blob/master/retinanet.py
        Freeze BatchNorm layers
        :return:
        """
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
