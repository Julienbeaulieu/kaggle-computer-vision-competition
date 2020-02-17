import torch
from typing import List, Dict, Union


class SoftMaxCE(torch.nn.Module):
    """
    Normal softmax cross entropy
    """

    def __init__(self, weights: Union[None, List[float]], ohem_rate: float):
        """

        :param weights: class weights
        """
        super(SoftMaxCE, self).__init__()

        self.ohem_rate = ohem_rate
        if weights is not None:
            self.loss_fn = torch.nn.CrossEntropyLoss(torch.tensor(weights), reduction='none')
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        losses = self.loss_fn(logits, labels)
        if self.ohem_rate < 1:
            loss = self.compute_ohem_loss(losses)
        else:
            loss = losses.mean()
        return loss

    def compute_ohem_loss(self, losses: torch.Tensor):
        N = losses.shape[0]
        keep_size = int(N * self.ohem_rate)
        ohem_losses, _ = losses.topk(keep_size)
        loss = ohem_losses.mean()
        return loss