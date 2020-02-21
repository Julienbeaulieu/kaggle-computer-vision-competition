import torch
import pickle
import numpy as np
from typing import List, Dict, Union
from yacs.config import CfgNode
from torch import nn
from sklearn.metrics.classification import classification_report

LOSS_FN = {
    'xentropy': torch.nn.CrossEntropyLoss
}


class EvalBlock(nn.Module):

    def __init__(self, solver_cfg: CfgNode, weights: Union[None, List[float]]):
        super(EvalBlock, self).__init__()
        loss_fn = solver_cfg.LOSS_FN 
        if weights is not None:
            self.loss_fn = LOSS_FN[loss_fn](torch.tensor(weights), reduction='none')
        else:
            self.loss_fn = LOSS_FN[loss_fn](reduction='none')
        self.ohem_rate = solver_cfg.OHEM_RATE

    def forward(self, logits, labels):
        losses = self.loss_fn(logits, labels) # CrossEntropyLoss -> takes xentropy(x, y)
        if self.ohem_rate < 1:
            loss = self.compute_ohem_loss(losses)
        else:
            loss = losses.mean()
        preds = torch.argmax(logits, dim=1)
        corrects = (labels == preds)
        acc = torch.sum(corrects) / (len(corrects) + 0.0)
        return loss, acc

    def compute_ohem_loss(self, losses: torch.Tensor):
        N = losses.shape[0]

        # What % of examples should we keep for our loss function? 
        keep_size = int(N*self.ohem_rate)
        # Get idx of top losses 
        _, ohem_indices = losses.topk(keep_size)
        ohem_losses = losses[ohem_indices]
        loss = ohem_losses.mean()
        return loss


class MultiHeadsEval(nn.Module):

    def __init__(self, solver_cfg: CfgNode):
        super(MultiHeadsEval, self).__init__()
        weights_path = solver_cfg.LABELS_WEIGHTS_PATH
        if weights_path != '':
            weights_data = pickle.load(open(weights_path, 'rb'))
            grapheme_weights = weights_data['grapheme']
            vowel_weights = weights_data['vowel']
            consonant_weights = weights_data['consonant']
        else:
            grapheme_weights = None
            vowel_weights = None
            consonant_weights = None
        loss_fn = solver_cfg.LOSS_FN
        self.grapheme_eval = EvalBlock(solver_cfg, grapheme_weights)
        self.vowel_eval = EvalBlock(solver_cfg, vowel_weights)
        self.consonant_eval = EvalBlock(solver_cfg, consonant_weights)
        self.grapheme_logits_cache = []
        self.vowel_logits_cache = []
        self.consonant_logits_cache = []
        self.labels_cache = []
        self.acc_cache = []
        self.loss_cache = []

    def forward(self, grapheme_logits: torch.Tensor, vowel_logits: torch.Tensor, consonant_logits: torch.Tensor,
                labels: torch.Tensor) -> Dict:
        # compute loss - call EvalBlock's forward function on our 3 classes
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
        # dump data in cache
        self.grapheme_logits_cache.append(grapheme_logits.detach().cpu().numpy())
        self.vowel_logits_cache.append(vowel_logits.detach().cpu().numpy())
        self.consonant_logits_cache.append(consonant_logits.detach().cpu().numpy())
        self.labels_cache.append(labels.detach().cpu().numpy())
        self.loss_cache.append(loss.detach().item())
        self.acc_cache.append(acc.detach().item())


        return eval_result

    def clear_cache(self):
        self.grapheme_logits_cache = []
        self.vowel_logits_cache = []
        self.consonant_logits_cache = []
        self.labels_cache = []
        self.loss_cache = []
        self.acc_cache = []


    def evalulate_on_cache(self):
        grapheme_logits_all = np.vstack(self.grapheme_logits_cache)
        vowel_logits_all = np.vstack(self.vowel_logits_cache)
        consonant_logits_all = np.vstack(self.consonant_logits_cache)
        labels_all = np.vstack(self.labels_cache)

        grapheme_preds = np.argmax(grapheme_logits_all, axis=1)
        vowels_preds = np.argmax(vowel_logits_all, axis=1)
        consonant_preds = np.argmax(consonant_logits_all, axis=1)

        grapheme_clf_result = classification_report(labels_all[:, 0], grapheme_preds, output_dict=True)
        vowels_clf_result = classification_report(labels_all[:, 1], vowels_preds, output_dict=True)
        consonant_clf_result = classification_report(labels_all[:, 2], consonant_preds, output_dict=True)
        kaggle_score = (grapheme_clf_result['macro avg']['recall'] * 2 + vowels_clf_result['macro avg']['recall'] +
                        consonant_clf_result['macro avg']['recall']) / 4


        acc = np.mean(self.acc_cache)
        loss = np.mean(self.loss_cache)


        result = {
            'grapheme_clf_result': grapheme_clf_result,
            'vowels_clf_result': vowels_clf_result,
            'consonant_clf_result': consonant_clf_result,
            'kaggle_score': kaggle_score,
            #'preds_labels': preds_labels,
            'acc': acc,
            'loss': loss

        }
        return result

def build_evaluator(solver_cfg: CfgNode) -> MultiHeadsEval:
    return MultiHeadsEval(solver_cfg)
