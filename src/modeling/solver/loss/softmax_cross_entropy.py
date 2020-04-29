import torch
from typing import List, Dict, Union
from yacs.config import CfgNode
from .jenson_shannon_divergence import jensen_shannon_divergence
from .build import LOSS_REGISTRY


@LOSS_REGISTRY.register('xentropy')
class SoftmaxCE(torch.nn.Module):
    """
    Normal softmax cross entropy
    """

    def __init__(self, loss_cfg: CfgNode, do_mixup: bool, weights: List, **kwargs):
        """
        :param weights: class weights
        """
        super(SoftmaxCE, self).__init__()

        self.ohem_rate = loss_cfg.OHEM_RATE
        self.do_mixup = do_mixup
        if weights is not None:
            self.loss_fn = torch.nn.CrossEntropyLoss(torch.tensor(weights), reduction='none')
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

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
            losses = self.loss_fn(logits, labels)
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
        loss = lam * self.loss_fn(logits, labels) + (1 - lam) * self.loss_fn(logits, shuffled_labels)
        return loss

    def compute_ohem_loss(self, losses: torch.Tensor):
        N = losses.shape[0]
        keep_size = int(N * self.ohem_rate)
        ohem_losses, _ = losses.topk(keep_size)
        loss = ohem_losses.mean()
        return loss
