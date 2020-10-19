import warnings
from typing import Callable, List, Optional, Tuple, Type, Union
from collections import defaultdict
import numpy as np
from src.reporter import DictReporter, StringReporter
from src.entity import EntityFromList
from src.scheme import IOB2

PER_CLASS_SCORES = Tuple[List[float], List[float], List[float], List[int]]
AVERAGE_SCORES = Tuple[float, float, float, int]
SCORES = Union[PER_CLASS_SCORES, AVERAGE_SCORES]


def _prf_divide(numerator, denominator, metric,
                modifier, average, warn_for, zero_division='warn'):
    """Performs division and handles divide-by-zero.
    On zero-division, sets the corresponding result elements equal to
    0 or 1 (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.
    The metric, modifier and average arguments are used only for determining
    an appropriate warning.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ['warn', 0] else 1.0

    # the user will be removing warnings if zero_division is set to something
    # different than its default value. If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != 'warn' or metric not in warn_for:
        return result

    # build appropriate warning
    # E.g. "Precision and F-score are ill-defined and being set to 0.0 in
    # labels with no predicted samples. Use ``zero_division`` parameter to
    # control this behavior."

    if metric in warn_for and 'f-score' in warn_for:
        msg_start = '{0} and F-score are'.format(metric.title())
    elif metric in warn_for:
        msg_start = '{0} is'.format(metric.title())
    elif 'f-score' in warn_for:
        msg_start = 'F-score is'
    else:
        return result

    _warn_prf(average, modifier, msg_start, len(result))

    return result


def _warn_prf(average, modifier, msg_start, result_size):
    axis0, axis1 = 'sample', 'label'
    if average == 'samples':
        axis0, axis1 = axis1, axis0
    msg = ('{0} ill-defined and being set to 0.0 {{0}} '
           'no {1} {2}s. Use `zero_division` parameter to control'
           ' this behavior.'.format(msg_start, modifier, axis0))
    if result_size == 1:
        msg = msg.format('due to')
    else:
        msg = msg.format('in {0}s with'.format(axis1))
    # warnings.warn(msg, UndefinedMetricWarning, stacklevel=2)


def check_consistent_length(y_true: List[List[str]], y_pred: List[List[str]]):
    """Check that all arrays have consistent first and second dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Args:
        y_true : 2d array.
        y_pred : 2d array.
    """
    len_true = list(map(len, y_true))
    len_pred = list(map(len, y_pred))
    is_list = set(map(type, y_true)) | set(map(type, y_pred))
    if not is_list == {list}:
        raise TypeError('Found input variables without list of list.')

    if len(y_true) != len(y_pred) or len_true != len_pred:
        message = 'Found input variables with inconsistent numbers of samples:\n{}\n{}'.format(len_true, len_pred)
        raise ValueError(message)


def _precision_recall_fscore_support(y_true: List[List[str]],
                                     y_pred: List[List[str]],
                                     *,
                                     average: Optional[str] = None,
                                     warn_for=('precision', 'recall', 'f-score'),
                                     beta: float = 1.0,
                                     sample_weight: Optional[List[int]] = None,
                                     zero_division: str = 'warn',
                                     scheme,
                                     suffix: bool = False,
                                     extract_tp_actual_correct: Callable = None) -> SCORES:
    if beta < 0:
        raise ValueError('beta should be >=0 in the F-beta score')

    average_options = (None, 'micro', 'macro', 'weighted')
    if average not in average_options:
        raise ValueError('average has to be one of {}'.format(average_options))

    check_consistent_length(y_true, y_pred)

    pred_sum, tp_sum, true_sum = extract_tp_actual_correct(y_true, y_pred, suffix, scheme)

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta ** 2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(
        numerator=tp_sum,
        denominator=pred_sum,
        metric='precision',
        modifier='predicted',
        average=average,
        warn_for=warn_for,
        zero_division=zero_division
    )
    recall = _prf_divide(
        numerator=tp_sum,
        denominator=true_sum,
        metric='recall',
        modifier='true',
        average=average,
        warn_for=warn_for,
        zero_division=zero_division
    )

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == 'warn' and ('f-score',) == warn_for:
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(
                average, 'true nor predicted', 'F-score is', len(true_sum)
            )

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall

        denom[denom == 0.] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    # Average the results
    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            zero_division_value = 0.0 if zero_division in ['warn', 0] else 1.0
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            return (zero_division_value if pred_sum.sum() == 0 else 0.0,
                    zero_division_value,
                    zero_division_value if pred_sum.sum() == 0 else 0.0,
                    sum(true_sum))

    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = sum(true_sum)

    return precision, recall, f_score, true_sum


def precision_recall_fscore_support(y_true: List[List[str]],
                                    y_pred: List[List[str]],
                                    *,
                                    average: Optional[str] = None,
                                    warn_for=('precision', 'recall', 'f-score'),
                                    beta: float = 1.0,
                                    sample_weight: Optional[List[int]] = None,
                                    zero_division: str = 'warn',
                                    scheme,
                                    suffix: bool = False,
                                    **kwargs) -> SCORES:
    def extract_tp_actual_correct(y_true, y_pred, suffix, scheme):
        # If this function is called from classification_report,
        # try to reuse entities to optimize the function.
        true_entities = set(EntityFromList(y_true, scheme, eval=True).entities)
        pred_entities = set(EntityFromList(y_pred, scheme, eval=True).entities)

        true_entities_dict = defaultdict(set)
        pred_entities_dict = defaultdict(set)
        for _, type, start_pos, end_pos, _ in true_entities:
            true_entities_dict[type].add((start_pos, end_pos))
        for _, type, start_pos, end_pos, _ in pred_entities:
            pred_entities_dict[type].add((start_pos, end_pos))
        
        target_type_names = sorted(set(true_entities_dict.keys()) | set(pred_entities_dict.keys()))
        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        for type in target_type_names:
            entities_true_type = true_entities_dict.get(type, set())
            entities_pred_type = pred_entities_dict.get(type, set())
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))

        return pred_sum, tp_sum, true_sum

    precision, recall, f_score, true_sum = _precision_recall_fscore_support(
        y_true, y_pred,
        average=average,
        warn_for=warn_for,
        beta=beta,
        sample_weight=sample_weight,
        zero_division=zero_division,
        scheme=scheme,
        suffix=suffix,
        extract_tp_actual_correct=extract_tp_actual_correct
    )

    return precision, recall, f_score, true_sum


def classification_report(y_true: List[List[str]],
                          y_pred: List[List[str]],
                          *,
                          sample_weight: Optional[List[int]] = None,
                          digits: int = 2,
                          output_dict: bool = False,
                          zero_division: str = 'warn',
                          suffix: bool = False,
                          scheme) -> Union[str, dict]:
    check_consistent_length(y_true, y_pred)

    # if scheme is None or not issubclass(scheme, Token):
    #     scheme = auto_detect(y_true, suffix)

    true_entities = set(EntityFromList(y_true, scheme, eval=True).entities)
    pred_entities = set(EntityFromList(y_pred, scheme, eval=True).entities)

    true_entities_dict = defaultdict(set)
    pred_entities_dict = defaultdict(set)
    for _, type, start_pos, end_pos, _ in true_entities:
        true_entities_dict[type].add((start_pos, end_pos))
    for _, type, start_pos, end_pos, _ in pred_entities:
        pred_entities_dict[type].add((start_pos, end_pos))
    
    target_type_names = sorted(set(true_entities_dict.keys()) | set(pred_entities_dict.keys()))

    if output_dict:
        reporter = DictReporter()
    else:
        name_width = max(map(len, target_type_names))
        avg_width = len('weighted avg')
        width = max(name_width, avg_width, digits)
        reporter = StringReporter(width=width, digits=digits)

    # compute per-class scores.
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred,
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
        scheme=scheme,
        suffix=suffix,
        entities_true=true_entities,
        entities_pred=pred_entities
    )
    for row in zip(target_type_names, p, r, f1, s):
        reporter.write(*row)
    reporter.write_blank()

    # compute average scores.
    average_options = ('micro', 'macro', 'weighted')
    for average in average_options:
        avg_p, avg_r, avg_f1, support = precision_recall_fscore_support(
            y_true, y_pred,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
            scheme=scheme,
            suffix=suffix,
            entities_true=true_entities,
            entities_pred=pred_entities
        )
        reporter.write('{} avg'.format(average), avg_p, avg_r, avg_f1, support)
    reporter.write_blank()

    return reporter.report()