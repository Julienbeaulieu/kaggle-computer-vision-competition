from yacs.config import CfgNode
from torch import nn
from src.tools.registry import Registry

LOSS_REGISTRY = Registry()


def build_loss(loss_cfg: CfgNode, **kwargs) -> nn.Module:
    loss_module = LOSS_REGISTRY[loss_cfg.NAME](loss_cfg, **kwargs)
    return loss_module

