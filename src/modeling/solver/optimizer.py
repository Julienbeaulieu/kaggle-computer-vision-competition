import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR
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
    opti_type = solver_cfg.OPTIMIZER
    lr = solver_cfg.BASE_LR
    return optimzers[opti_type](parameters, lr=lr)


def build_scheduler(optimizer: torch.optim, solver_cfg: CfgNode):
    """
    CyclicLR: https://arxiv.org/abs/1506.01186
    """
    base_lr = solver_cfg.BASE_LR
    max_lr = solver_cfg.MAX_LR
    step_size_up = solver_cfg.STEP_SIZE_UP
    mode = solver_cfg.MODE 
    scheduler = CyclicLR(optimizer, base_lr, max_lr, step_size_up=step_size_up, mode=mode, cycle_momentum=False)
    return scheduler