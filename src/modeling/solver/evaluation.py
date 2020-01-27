import torch
import pickle
from typing import List
from yacs.config import CfgNode
from torch import nn

LOSS_FN = {
    'xentropy': torch.nn.CrossEntropyLoss
}


class EvalBlock(nn.Module):

    def __init__(self, loss_fn: str, weights: List[float]):
        super(EvalBlock, self).__init__()
        self.loss_fn = LOSS_FN[loss_fn](torch.tensor(weights))

    def forward(self, logits, labels):
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        corrects = (labels == preds)
        acc = torch.sum(corrects) / (len(corrects) + 0.0)
        return loss, acc


class MultiHeadsEval(nn.Module):

    def __init__(self, solver_cfg: CfgNode):
        super(MultiHeadsEval, self).__init__()
        weights_path = solver_cfg.LABELS_WEIGHTS_PATH
        weights_data = pickle.load(open(weights_path, 'rb'))
        grapheme_weights = weights_data['grapheme']
        vowel_weights = weights_data['vowel']
        consonant_weights = weights_data['consonant']
        loss_fn = solver_cfg.LOSS_FN
        self.grapheme_eval = EvalBlock(loss_fn, grapheme_weights)
        self.vowel_eval = EvalBlock(loss_fn, vowel_weights)
        self.consonant_eval = EvalBlock(loss_fn, consonant_weights)

    def forward(self, grapheme_logits, vowel_logits, consonant_logits, labels):
        grapheme_loss, grapheme_acc = self.grapheme_eval(grapheme_logits, labels[:, 0])
        vowel_loss, vowel_acc = self.vowel_eval(vowel_logits, labels[:, 1])
        consonant_loss, consonant_acc = self.consonant_eval(consonant_logits, labels[:, 2])
        loss = grapheme_loss + vowel_loss + consonant_loss
        acc = (grapheme_acc + vowel_acc + consonant_acc) / 3
        eval_result = {
            'grapheme_loss': grapheme_loss,
            'grapheme_acc': grapheme_acc,
            'vowel_loss': vowel_loss,
            'vowel_acc': vowel_acc,
            'consonant_loss': consonant_loss,
            'consonant_acc': consonant_acc,
            'loss': loss,
            'acc': acc
        }
        return eval_result


def build_evaluator(solver_cfg: CfgNode) -> nn.Module:
    return MultiHeadsEval(solver_cfg)
