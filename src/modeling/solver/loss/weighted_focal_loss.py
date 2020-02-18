import torch
from torch.nn import functional as F
from typing import List, Dict, Union
from yacs.config import CfgNode
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss


class WeightedFocalLoss(torch.nn.Module):
    """
    If dysfunctional, likely incorrect implementation
    """

    def __init__(self, num_classes, class_weights, gamma=1):
        """

        :param num_classes: number of classes
        :param class_weights: default class weights
        """
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = torch.nn.Parameter(torch.tensor(class_weights), requires_grad=False)
        self.dummy_eyes = torch.nn.Parameter(torch.eye(num_classes), requires_grad=False)
        self.gamma = gamma

    def forward(self, logits, labels):
        targets = self.dummy_eyes[labels]
        alpha = self.class_weights[labels]

        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        #  https://github.com/richardaecn/class-balanced-loss/blob/master/src/cifar_main.py#L226-L266
        #  The below is SUPPOSED TO BE EQUIVALENT TO
        #  p = logits.sigmoid()
        #  pt = p * targets + (1 - p) * (1 - targets)
        #  modulator = (1 - pt)**gamma
        # I don't know how,  I don't know why
        modulator = torch.exp(
            -self.gamma * targets * logits - self.gamma * torch.log1p(torch.exp(-1 * logits)))

        loss = modulator * ce_loss
        loss = loss.sum(dim=1)
        loss = loss * alpha
        avg_loss = loss.mean()

        return avg_loss
