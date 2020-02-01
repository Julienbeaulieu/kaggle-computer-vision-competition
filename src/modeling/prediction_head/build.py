from yacs.config import CfgNode
from torch import nn
from src.tools.registry import Registry

HEAD_REGISTRY = Registry()


def build_head(head_cfg: CfgNode) -> nn.Module:
    head_module = HEAD_REGISTRY[head_cfg.NAME](head_cfg)
    return head_module
