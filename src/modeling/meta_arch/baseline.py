import torch
from torch import nn
from yacs.config import CfgNode
from src.modeling.backbone.build import build_backbone
from src.modeling.prediction_head.build import build_head

from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register('baseline')
def build_baseline_model(model_cfg: CfgNode) -> nn.Module:
    """
    Builds the baseline model using the CFGNode object,
    :param model_cfg: YAML based YACS configuration node.
    :return: return torch neural network module
    """
    # instantiate and return the BaselineModel using the configuration node
    return BaselineModel(model_cfg)


# Baseline model class must extends torch.nn.module
class BaselineModel(nn.Module):

    def __init__(self, model_cfg: CfgNode):
        """
        # Constructor will take a CFG node to read key properties from it.
        :param model_cfg:
        """
        # Cell the super constructor
        super(BaselineModel, self).__init__()

        # Build backbone using backbone dict of property
        self.backbone = build_backbone(model_cfg.BACKBONE)

        # Build backbone using head dict of properties
        self.head = build_head(model_cfg.HEAD)

        # Store output dimensions in the base model instance variables
        self.heads_dims = model_cfg.HEAD.OUTPUT_DIMS

    def forward(self, x):
        """
        Build backbone, build head, return it.
        :param x:
        :return:
        """
        # Call build_backbone function on input
        x = self.backbone(x)

        # Call build_head function on the output above
        x = self.head(x)
        # split output into the 3 classes
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
