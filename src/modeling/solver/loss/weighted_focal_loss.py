import torch
from typing import List
from yacs.config import CfgNode
from torch.nn import functional as F
from .jenson_shannon_divergence import jensen_shannon_divergence
from .build import LOSS_REGISTRY


@LOSS_REGISTRY.register('weighted_focal_loss')
class WeightedFocalLoss(torch.nn.Module):
    """
    If dysfunctional, likely incorrect implementation
    """

    def __init__(self, loss_cfg: CfgNode, num_classes: int, weights: List, **kwargs):
        """

        :param num_classes: number of classes
        :param weights: default class weights
        :param gamma: gamma for focal loss
        """
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = torch.nn.Parameter(torch.tensor(weights), requires_grad=False)
        self.dummy_eyes = torch.nn.Parameter(torch.eye(num_classes), requires_grad=False)
        focal_loss_cfg = loss_cfg.FOCAL_LOSS
        self.gamma = focal_loss_cfg.GAMMA

    def forward(self, logits, labels, js_divergence=False):

        if js_divergence:
            logits, logits_aug1, logits_aug2 = torch.chunk(logits, 3, dim=0)
            loss = jensen_shannon_divergence(logits, logits_aug1, logits_aug2)
        else:
            loss = 0

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

        preds = torch.argmax(logits.float(), dim=1)
        corrects = (labels == preds)
        acc = torch.sum(corrects) / (len(corrects) + 0.0)

        return avg_loss, acc
