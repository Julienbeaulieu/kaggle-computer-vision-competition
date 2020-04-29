import torch
import pickle
import numpy as np
from torch import nn
from collections import Counter
from yacs.config import CfgNode
from typing import List, Dict, Union
from sklearn.metrics import classification_report
#from .loss import WeightedFocalLoss, SoftMaxCE
from .loss import build_loss

class MultiHeadsEvaluation(nn.Module):
    def __init__(self, solver_cfg: CfgNode):
        super(MultiHeadsEvaluation, self).__init__()
        loss_cfg = solver_cfg.LOSS
        weights_path = solver_cfg.LABELS_WEIGHTS_PATH
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

        self.grapheme_loss_fn = build_loss(loss_cfg, do_mixup=do_mixup, weights=grapheme_weights, eps=loss_cfg.EPS, reduction=loss_cfg.REDUCTION, num_classes=168)
        self.vowel_loss_fn = build_loss(loss_cfg, do_mixup=do_mixup, weights=vowel_weights, eps=loss_cfg.EPS, reduction=loss_cfg.REDUCTION, num_classes=11)
        self.consonant_loss_fn = build_loss(loss_cfg, do_mixup=do_mixup, weights=consonant_weights, eps=loss_cfg.EPS, reduction=loss_cfg.REDUCTION, num_classes=7)
        self.do_mixup = do_mixup

        self.grapheme_logits_cache = []
        self.vowel_logits_cache = []
        self.consonant_logits_cache = []
        self.labels_cache = []
        self.acc_cache = []
        self.loss_cache = []

    def forward(self, grapheme_logits: torch.Tensor, vowel_logits: torch.Tensor, consonant_logits: torch.Tensor,
                labels: Union[tuple, torch.Tensor], js_divergence: bool = False) -> Dict:

        if self.do_mixup:
            labels, grapheme_labels, vowel_labels, consonant_labels = mixup_labels_helper(labels)
        else:
            grapheme_labels, vowel_labels, consonant_labels = labels[:, 0], labels[:, 1], labels[:, 2]

        grapheme_loss, grapheme_acc = self.grapheme_loss_fn(grapheme_logits, grapheme_labels, js_divergence)
        vowel_loss, vowel_acc = self.vowel_loss_fn(vowel_logits, vowel_labels, js_divergence)
        consonant_loss, consonant_acc = self.consonant_loss_fn(consonant_logits, consonant_labels, js_divergence)

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

        preds_labels = []
        for idx, grapheme_pred in enumerate(grapheme_preds):
            vowel_pred = vowels_preds[idx]
            consonant_pred = consonant_preds[idx]
            labels = labels_all[idx]
            entry = {
                'grapheme_pred': grapheme_pred,
                'vowel_pred': vowel_pred,
                'consonant_pred': consonant_pred,
                'grapheme_label': labels[0],
                'vowel_label': labels[1],
                'consonant_label': labels[2]
            }
            preds_labels.append(entry)

        grapheme_clf_result = clf_result_helper(grapheme_clf_result, preds_labels, 'grapheme_pred', 'grapheme_label')
        vowels_clf_result = clf_result_helper(vowels_clf_result, preds_labels, 'vowel_pred', 'vowel_label')
        consonant_clf_result = clf_result_helper(consonant_clf_result, preds_labels, 'consonant_pred',
                                                 'consonant_label')

        acc = np.mean(self.acc_cache)
        loss = np.mean(self.loss_cache)
        result = {
            'grapheme_clf_result': grapheme_clf_result,
            'vowel_clf_result': vowels_clf_result,
            'consonant_clf_result': consonant_clf_result,
            'kaggle_score': kaggle_score,
            'preds_labels': preds_labels,
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


def clf_result_helper(clf_result: Dict, preds_labels: List, pred_key: str, label_key: str):
    """
    a helper function get per class result the highest error class, and highest error class occurences
    :param clf_result:  classfier result dict from classificaiton_report
    :param preds_labels: list of preds and labels
    :param pred_key:  one of [grapheme_pred, vowel_pred, consonant_pred]
    :param label_key: one of [grapheme_label, vowel_label, consonant_label]
    :return: list view of clf result with some added info
    """
    for k in clf_result.keys():
        if k not in ['accuracy', 'macro avg', 'weighted avg']:
            cls = int(k)
            preds_counts = Counter([x[pred_key] for x in preds_labels if x[label_key] == cls])
            preds_counts = [[k, preds_counts[k]] for k in preds_counts]
            incorrect_preds_counts = [x for x in preds_counts if x[0] != cls]
            if len(incorrect_preds_counts) > 0:
                highest_error_cls, highest_error_cls_num = sorted(incorrect_preds_counts, key=lambda x: x[1])[-1]
            else:
                highest_error_cls, highest_error_cls_num = -1, 0

            clf_result[k]['class'] = cls
            clf_result[k]['error_cls'] = highest_error_cls
            if clf_result[k]['support'] > 0:
                clf_result[k]['error_cls_rate'] = highest_error_cls_num / clf_result[k]['support']
            else:
                clf_result[k]['error_cls_rate'] = 0
    clf_result = [clf_result[k] for k in clf_result if k not in ['accuracy', 'macro avg', 'weighted avg']]
    return clf_result


def build_evaluator(solver_cfg: CfgNode):
    if solver_cfg.MIXUP_AUGMENT:
        nomixup_cfg = solver_cfg.clone()
        nomixup_cfg.MIXUP_AUGMENT = False
        return MultiHeadsEvaluation(nomixup_cfg), MultiHeadsEvaluation(solver_cfg)
    else:
        return MultiHeadsEvaluation(solver_cfg), None
