import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from yacs.config import CfgNode

def build_optimizer(model: torch.nn.Module, opti_cfg: CfgNode) -> Optimizer:
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
    opti_type = opti_cfg.NAME
    lr = opti_cfg.BASE_LR
    return optimzers[opti_type](parameters, lr=lr)

def build_scheduler(optimizer: torch.optim, scheduler_cfg: CfgNode, epochs, steps_per_epoch):
    """
    OneCycleLR:    https://arxiv.org/abs/1708.07120
    """
    scheduler_type = scheduler_cfg.NAME
    if scheduler_type == 'unchange':
        return None    
    elif scheduler_type == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer, 
                               max_lr=scheduler_cfg.MAX_LR, 
                               steps_per_epoch=steps_per_epoch,
                               epochs=epochs,
                               pct_start=scheduler_cfg.PCT_START,
                               anneal_strategy=scheduler_cfg.ANNEAL_STRATEGY, 
                               div_factor=scheduler_cfg.DIV_FACTOR,
                               cycle_momentum=True)
        return scheduler
    elif scheduler_type == 'ReduceLROnPlateau':
        gamma = scheduler_cfg.LR_REDUCE_GAMMA
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=gamma)
        return scheduler
    else:
        raise Exception('scheduler name invalid, choices are: unchange/OneCycleLR/ReduceLROnPlatea')