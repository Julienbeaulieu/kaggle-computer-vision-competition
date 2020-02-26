from torch import nn
from yacs.config import CfgNode
from src.tools.registry import Registry
from src.modeling.layers.sync_batchnorm import convert_model
META_ARCH_REGISTRY = Registry()


def build_model(model_cfg: CfgNode) -> nn.Module:
    """
    build model
    :param model_cfg: model config blob
    :return: model
    """
    meta_arch = model_cfg.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(model_cfg)

    # This is VERY SLOW
    if model_cfg.NORMALIZATION_FN == 'SYNC_BN':
        model = convert_model(model)

    return model
