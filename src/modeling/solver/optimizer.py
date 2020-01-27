import torch
from torch.optim.optimizer import Optimizer
from yacs.config import CfgNode


def build_optimizer(model: torch.nn.Module, solver_cfg: CfgNode) -> Optimizer:
    """
    simple optimizer builder
    :param model: already gpu pushed model
    :param solver_cfg:  config node
    :return: the optimizer
    """
    parameters = model.parameters()
    optimzers = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD
    }
    opti_type = solver_cfg.TYPE
    lr = solver_cfg.BASE_LR
    return optimzers[opti_type](parameters, lr=lr)
