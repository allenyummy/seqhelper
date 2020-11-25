# encoding = utf-8
# Author: Yu-Lun Chiang
# Description: evaluation for sequence labeling

import logging
from typing import List, Tuple
from src.entity import EntityFromList
from src.scheme import IOB2

logger = logging.getLogger(__name__)


def f1_score(y_true, y_pred, scheme, average="micro"):
    """Compute F1 score."""
    nb_correct, nb_pred, nb_true = _calculate(y_true, y_pred, scheme)
    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    return score


def precision_score(y_true, y_pred, scheme, average="micro"):
    """Compute precision score."""
    nb_correct, nb_pred, _ = _calculate(y_true, y_pred, scheme)
    score = nb_correct / nb_pred if nb_pred > 0 else 0
    return score


def recall_score(y_true, y_pred, scheme, average="micro"):
    """Compute recall score."""
    nb_correct, _, nb_true = _calculate(y_true, y_pred, scheme)
    score = nb_correct / nb_true if nb_true > 0 else 0
    return score


def accuracy_score(y_true, y_pred, scheme):
    """Compute accuracy score."""
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]
    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)
    score = nb_correct / nb_true
    return score


def _calculate(y_true, y_pred, scheme):
    true_entities = set(EntityFromList(y_true, scheme, eval=True).entities)
    pred_entities = set(EntityFromList(y_pred, scheme, eval=True).entities)
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)
    return nb_correct, nb_pred, nb_true