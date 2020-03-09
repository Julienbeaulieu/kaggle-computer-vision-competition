import torch
import pickle
import numpy as np
from typing import List, Dict, Union
from yacs.config import CfgNode
from torch import nn
from sklearn.metrics import classification_report
from .loss import build_loss

class MultiHeadsEval(nn.Module):

    def __init__(self, solver_cfg: CfgNode):
        super(MultiHeadsEval, self).__init__()
        weights_path = solver_cfg.LABELS_WEIGHTS_PATH
        loss_cfg = solver_cfg.LOSS
        do_mixup = solver_cfg.MIXUP_AUGMENT
        if weights_path != '':
            weights_data = pickle.load(open(weights_path, 'rb'))
            grapheme_weights = weights_data['grapheme']
            vowel_weights = weights_data['vowel']
            consonant_weights = weights_data['consonant']
        else:
            grapheme_weights = None
            vowel_weights = None
            consonant_weights = None

        self.grapheme_loss_fn = build_loss(loss_cfg, do_mixup=do_mixup, weights=grapheme_weights, num_classes=168)
        self.vowel_loss_fn = build_loss(loss_cfg, do_mixup=do_mixup, weights=vowel_weights, num_classes=11)
        self.consonant_loss_fn = build_loss(loss_cfg, do_mixup=do_mixup, weights=consonant_weights, num_classes=7)
        self.do_mixup = do_mixup
        
        self.grapheme_logits_cache = []
        self.vowel_logits_cache = []
        self.consonant_logits_cache = []
        self.labels_cache = []
        self.acc_cache = []
        self.loss_cache = []

    def forward(self, grapheme_logits: torch.Tensor, vowel_logits: torch.Tensor, consonant_logits: torch.Tensor,
                labels: torch.Tensor) -> Dict:

        if self.do_mixup:
            labels, grapheme_labels, vowel_labels, consonant_labels = mixup_labels_helper(labels)
        else:
            grapheme_labels, vowel_labels, consonant_labels = labels[:, 0], labels[:, 1], labels[:, 2]

        # compute loss - call EvalBlock's forward function on our 3 classes
        grapheme_loss, grapheme_acc = self.grapheme_loss_fn(grapheme_logits, grapheme_labels)
        vowel_loss, vowel_acc = self.vowel_loss_fn(vowel_logits, vowel_labels)
        consonant_loss, consonant_acc = self.consonant_loss_fn(consonant_logits, consonant_labels)

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


    def evaluate_on_cache(self):
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

def mixup_labels_helper(labels: tuple):
    grapheme_labels, shuffled_grapheme_labels, vowel_labels, shuffled_vowel_labels, \
    consonant_labels, shuffled_consonant_labels, lam = labels
    labels = torch.stack([grapheme_labels, vowel_labels, consonant_labels]).transpose(0, 1)

    grapheme_labels = (grapheme_labels, shuffled_grapheme_labels, lam)
    vowel_labels = (vowel_labels, shuffled_vowel_labels, lam)
    consonant_labels = (consonant_labels, shuffled_consonant_labels, lam)
    return labels, grapheme_labels, vowel_labels, consonant_labels

def build_evaluator(solver_cfg: CfgNode) -> MultiHeadsEval:
    if solver_cfg.MIXUP_AUGMENT:
        nomixup_cfg = solver_cfg.clone()
        nomixup_cfg.MIXUP_AUGMENT = False
        return MultiHeadsEval(nomixup_cfg), MultiHeadsEval(solver_cfg)
    else:
        return MultiHeadsEval(solver_cfg), None
