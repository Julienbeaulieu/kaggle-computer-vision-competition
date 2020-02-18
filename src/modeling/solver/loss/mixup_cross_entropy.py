import torch
from typing import List, Dict, Union


class MixUpCrossEntropy(torch.nn.Module):

    def __init__(self, ohem_rate: float):
        super(MixUpCrossEntropy, self).__init__()
        self.ohem_rate = ohem_rate
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, logits: torch.Tensor, mixedup_labels_data: tuple):
        """

        :param logits: computed logits
        :param mixedup_labels_data:
        :return:
        """
        labels, shuffled_labels, lam = mixedup_labels_data
        loss = lam * self.loss_fn(logits, labels) + (1 - lam) * self.loss_fn(logits, shuffled_labels)
        return loss
