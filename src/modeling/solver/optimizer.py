import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR
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

def build_scheduler(optimizer: torch.optim, solver_cfg: CfgNode, steps_per_epoch):
    """
    OneCycleLR:    https://arxiv.org/abs/1708.07120
    """
    scheduler = OneCycleLR(optimizer, 
                           max_lr=solver_cfg.MAX_LR, 
                           steps_per_epoch=steps_per_epoch,
                           epochs=solver_cfg.TOTAL_EPOCHS,
                           anneal_strategy='cos', 
                           cycle_momentum=True)
    return scheduler