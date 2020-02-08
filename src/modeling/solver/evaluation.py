import torch
import pickle
import numpy as np
from typing import List, Dict
from yacs.config import CfgNode
from torch import nn
from sklearn.metrics.classification import classification_report
from collections import Counter

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
        self.grapheme_logits_cache = []
        self.vowel_logits_cache = []
        self.consonant_logits_cache = []
        self.labels_cache = []

    def forward(self, grapheme_logits: torch.Tensor, vowel_logits: torch.Tensor, consonant_logits: torch.Tensor,
                labels: torch.Tensor) -> Dict:
        # compute loss
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

        return eval_result

    def clear_cache(self):
        self.grapheme_logits_cache = []
        self.vowel_logits_cache = []
        self.consonant_logits_cache = []
        self.labels_cache = []

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
                'consonant_label': labels[2],
            }
            preds_labels.append(entry)

        grapheme_clf_result = clf_result_helper(grapheme_clf_result, preds_labels, 'grapheme_pred', 'grapheme_label')
        vowels_clf_result = clf_result_helper(vowels_clf_result, preds_labels, 'vowel_pred', 'vowel_label')
        consonant_clf_result = clf_result_helper(consonant_clf_result, preds_labels, 'consonant_pred',
                                                 'consonant_label')

        result = {
            'grapheme_clf_result': grapheme_clf_result,
            'vowel_clf_result': vowels_clf_result,
            'consonant_clf_result': consonant_clf_result,
            'kaggle_score': kaggle_score
        }
        return result


def build_evaluator(solver_cfg: CfgNode) -> nn.Module:
    return MultiHeadsEval(solver_cfg)


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
            clf_result[k]['error_cls_rate'] = highest_error_cls_num / clf_result[k]['support']

    clf_result = [clf_result[k] for k in clf_result if k not in ['accuracy', 'macro avg', 'weighted avg']]
    return clf_result
