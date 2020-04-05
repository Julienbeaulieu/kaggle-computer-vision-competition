import random
import torch
import numpy as np
from yacs.config import CfgNode

"""
Code related to cutmix / mixup 
"""


class MixupAugmenter(torch.nn.Module):

    def __init__(self, mixup_cfg: CfgNode):
        super(MixupAugmenter, self).__init__()
        self.cutmix_alpha = mixup_cfg.CUTMIX_ALPHA
        self.mixup_alpha = mixup_cfg.MIXUP_ALPHA
        self.cutmix_prob = mixup_cfg.CUTMIX_PROB
    def forward(self, data, labels):
        do_cutmix = (random.random() <= self.cutmix_prob)
        if do_cutmix:
            alpha = self.cutmix_alpha
        else:
            alpha = self.mixup_alpha
        data, targets = mixup(data, labels, alpha, do_cutmix)
        return data, targets


def mixup(data, labels, alpha=1.0, cut_mix=False):
    targets1 = labels[:, 0]
    targets2 = labels[:, 1]
    targets3 = labels[:, 2]
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    if cut_mix:
        bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    else:
        data = data * lam + shuffled_data * (1 - lam)
    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]

    return data, targets


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
