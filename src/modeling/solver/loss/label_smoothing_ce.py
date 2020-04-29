import torch  
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union
from yacs.config import CfgNode
from .jenson_shannon_divergence import jensen_shannon_divergence
from .build import LOSS_REGISTRY


class LabelSmoothingCrossEntropy(torch.nn.Module):
    """
    Code copied from fastai2

    """
    y_int = True
    def __init__(self, loss_cfg:CfgNode, eps: float, reduction): 
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = loss_cfg.EPS
        self.reduction = loss_cfg.REDUCTION

    def forward(self, output, target, weights):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum': loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1) #We divide by that size at the return line so sum and not mean
            if self.reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target.long(), reduction=self.reduction)


@LOSS_REGISTRY.register('label_smoothing_ce')
class LabSmoothSoftmaxCE(torch.nn.Module):
    """
    Normal softmax cross entropy with added functionality
    """

    def __init__(self, loss_cfg: CfgNode, do_mixup: bool, weights: List, eps: float, reduction, **kwargs):
        """
        :param weights: class weights
        """
        super(LabSmoothSoftmaxCE, self).__init__()
        self.ohem_rate = loss_cfg.OHEM_RATE
        self.do_mixup = do_mixup
        self.eps = loss_cfg.EPS
        self.reduction = loss_cfg.REDUCTION
        self.weights = weights
        self.loss_fn = LabelSmoothingCrossEntropy(loss_cfg, self.eps, self.reduction)
        #self.loss_fn = CrossEntropy()

    def forward(self, logits, labels, js_divergence=False):
        
        if js_divergence:
            logits, logits_aug1, logits_aug2 = torch.chunk(logits, 3, dim=0)
            loss = jensen_shannon_divergence(logits, logits_aug1, logits_aug2)
        else:
            loss = 0

        preds = torch.argmax(logits.float(), dim=1)
        if self.do_mixup:
            losses = self.compute_mixup_loss(logits, labels)
            corrects = (labels[0] == preds)
        else:
            losses = self.loss_fn(logits, labels, self.weights)
            corrects = (labels == preds)

        if self.ohem_rate < 1:
            loss += self.compute_ohem_loss(losses)
        else:
            loss += losses.mean()

        acc = torch.sum(corrects) / (len(corrects) + 0.0)

        return loss, acc

    def compute_mixup_loss(self, logits: torch.Tensor, mixedup_labels_data: tuple):
        """

        :param logits: computed logits
        :param mixedup_labels_data:
        :return: 
        """
        labels, shuffled_labels, lam = mixedup_labels_data
        if self.weights is not None:
            loss = lam * self.loss_fn(logits, labels, torch.tensor(self.weights)) + (1 - lam) * self.loss_fn(logits, shuffled_labels, torch.tensor(self.weights))
        else:
            loss = lam * self.loss_fn(logits, labels) + (1 - lam) * self.loss_fn(logits, shuffled_labels)
        return loss

    def compute_ohem_loss(self, losses: torch.Tensor):
        N = losses.shape[0]
        keep_size = int(N * self.ohem_rate)
        ohem_losses, _ = losses.topk(keep_size)
        loss = ohem_losses.mean()
        return loss

