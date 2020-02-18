import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from yacs.config import CfgNode
from typing import Union


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
    opti_type = solver_cfg.OPTIMIZER
    lr = solver_cfg.BASE_LR
    return optimzers[opti_type](parameters, lr=lr)


def build_scheduler(optimizer: Optimizer, solver_cfg: CfgNode) -> Union[MultiStepLR, None]:
    """

    :param optimizer:
    :param optimizer: Optimizer
    :param solver_cfg:
    "param solver_cfg: CfgNode
    :return:
    """
    if len(solver_cfg.MULTI_STEPS_LR_MILESTONES) == 0:
        return None
    gamma = solver_cfg.LR_REDUCE_GAMMA
    milestones = solver_cfg.MULTI_STEPS_LR_MILESTONES
    scheduler = MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=-1)
    return scheduler
