# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import re
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple

from swift.plugin.metric import metric_mapping
from swift.utils import Serializer


def _wrap_collator_padding_to_compat(collator):
    try:
        signature = inspect.signature(collator)
    except (TypeError, ValueError):
        return None

    if 'padding_to' in signature.parameters:
        return None
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return None

    def wrapped_collator(batch, *args, padding_to=None, **kwargs):
        return collator(batch, *args, **kwargs)

    return wrapped_collator


def _patch_generate_context_remove_unused_columns() -> None:
    from swift.llm.template.base import Template

    if getattr(Template, '_lateral_decision_generate_context_patched', False):
        return

    origin_generate_context = Template.generate_context

    @contextmanager
    def generate_context(self):
        remove_unused_columns = self.remove_unused_columns
        data_collator = getattr(self, '_data_collator', None)
        wrapped_data_collator = _wrap_collator_padding_to_compat(data_collator)
        with origin_generate_context(self):
            self.remove_unused_columns = True
            if wrapped_data_collator is not None:
                self._data_collator = wrapped_data_collator
            try:
                yield
            finally:
                self.remove_unused_columns = remove_unused_columns
                if wrapped_data_collator is not None:
                    self._data_collator = data_collator

    Template.generate_context = generate_context
    Template._lateral_decision_generate_context_patched = True


_patch_generate_context_remove_unused_columns()


LATERAL_DECISION_NAME_TO_TOKEN = {
    '左换道': 'LAT_LANE_CHANGE_LEFT',
    '右换道': 'LAT_LANE_CHANGE_RIGHT',
    '左避让': 'LAT_NUDGE_LEFT',
    '右避让': 'LAT_NUDGE_RIGHT',
    '车道居中': 'LAT_LANE_KEEP',
    '车道内居中': 'LAT_LANE_KEEP',
    '路口顺行': 'LAT_INTERSECTION_FOLLOW',
    '路口左转': 'LAT_TURN_LEFT',
    '路口右转': 'LAT_TURN_RIGHT',
    '路口掉头': 'LAT_U_TURN',
}

LATERAL_DECISION_TOKENS = [
    'LAT_LANE_CHANGE_LEFT',
    'LAT_LANE_CHANGE_RIGHT',
    'LAT_NUDGE_LEFT',
    'LAT_NUDGE_RIGHT',
    'LAT_LANE_KEEP',
    'LAT_INTERSECTION_FOLLOW',
    'LAT_TURN_LEFT',
    'LAT_TURN_RIGHT',
    'LAT_U_TURN',
]

LONGITUDINAL_DECISION_NAME_TO_TOKEN = {
    '保持': 'LON_MAINTAIN',
    '加速': 'LON_ACCELERATE',
    '减速': 'LON_DECELERATE',
    '停车': 'LON_STOP',
}

LONGITUDINAL_DECISION_TOKENS = [
    'LON_MAINTAIN',
    'LON_ACCELERATE',
    'LON_DECELERATE',
    'LON_STOP',
]


def _decode_serialized_text(row) -> str:
    try:
        return Serializer.from_tensor(row)
    except Exception:
        return str(row)


def _normalize_lateral_decision(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip().strip('<>').strip()
    if value in LATERAL_DECISION_TOKENS:
        return value
    return LATERAL_DECISION_NAME_TO_TOKEN.get(value)


def _normalize_longitudinal_decision(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip().strip('<>').strip()
    if value in LONGITUDINAL_DECISION_TOKENS:
        return value
    return LONGITUDINAL_DECISION_NAME_TO_TOKEN.get(value)


def _extract_tokens(text: str) -> List[str]:
    return [token.strip() for token in re.findall(r'<([^<>]+)>', text or '')]


def extract_decisions_from_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    if '</think>' in response:
        response = response.split('</think>', 1)[1]

    lateral_decision = None
    longitudinal_decision = None
    for token in _extract_tokens(response):
        if lateral_decision is None:
            lateral_decision = _normalize_lateral_decision(token)
        if longitudinal_decision is None:
            longitudinal_decision = _normalize_longitudinal_decision(token)
        if lateral_decision is not None and longitudinal_decision is not None:
            break
    return lateral_decision, longitudinal_decision


def extract_decision_gt_from_labels(labels: str) -> Tuple[Set[str], Set[str]]:
    answer_match = re.search(r'<answer>(.*?)</answer>', labels or '', re.DOTALL)
    answer_content = answer_match.group(1) if answer_match else labels

    lateral_gt_set = set()
    longitudinal_gt_set = set()
    for result in answer_content.split(';'):
        tokens = _extract_tokens(result)
        if not tokens:
            continue
        lateral = _normalize_lateral_decision(tokens[0])
        if lateral is not None:
            lateral_gt_set.add(lateral)
        longitudinal = _normalize_longitudinal_decision(tokens[1] if len(tokens) >= 2 else None)
        if longitudinal is not None:
            longitudinal_gt_set.add(longitudinal)
    return lateral_gt_set, longitudinal_gt_set


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _init_class_counts(tokens: List[str]) -> Dict[str, Dict[str, int]]:
    return {token: {'tp': 0, 'fp': 0, 'fn': 0} for token in tokens}


def _update_class_counts(pred: Optional[str], gt_set: Set[str], class_counts: Dict[str, Dict[str, int]]) -> None:
    for token in class_counts:
        if pred == token and token in gt_set:
            class_counts[token]['tp'] += 1
        elif pred == token and token not in gt_set:
            class_counts[token]['fp'] += 1
        elif pred != token and token in gt_set:
            class_counts[token]['fn'] += 1


def _add_class_metrics(metrics: Dict[str, float], class_counts: Dict[str, Dict[str, int]]) -> None:
    for token, counts in class_counts.items():
        prefix = token.lower()
        precision = _safe_div(counts['tp'], counts['tp'] + counts['fp'])
        recall = _safe_div(counts['tp'], counts['tp'] + counts['fn'])
        f1 = _safe_div(2 * precision * recall, precision + recall)
        metrics[f'{prefix}_precision'] = precision
        metrics[f'{prefix}_recall'] = recall
        metrics[f'{prefix}_f1'] = f1


def _compute_decision_metrics(prediction, *, dimension: str, full: bool) -> Dict[str, float]:
    preds, labels = prediction[0], prediction[1]
    if dimension == 'lateral':
        tokens = LATERAL_DECISION_TOKENS
        metric_prefix = 'lateral'
    elif dimension == 'longitudinal':
        tokens = LONGITUDINAL_DECISION_TOKENS
        metric_prefix = 'longitudinal'
    else:
        raise ValueError(f'Unsupported decision dimension: {dimension}')

    class_counts = _init_class_counts(tokens)
    total = 0
    correct = 0

    for i in range(preds.shape[0]):
        pred_text = _decode_serialized_text(preds[i])
        label_text = _decode_serialized_text(labels[i])

        pred_lateral, pred_longitudinal = extract_decisions_from_response(pred_text)
        gt_lateral_set, gt_longitudinal_set = extract_decision_gt_from_labels(label_text)
        if dimension == 'lateral':
            pred_decision = pred_lateral
            gt_set = gt_lateral_set
        else:
            pred_decision = pred_longitudinal
            gt_set = gt_longitudinal_set

        if not gt_set:
            continue

        total += 1
        correct += int(pred_decision in gt_set)
        if full:
            _update_class_counts(pred_decision, gt_set, class_counts)

    metrics = {f'{metric_prefix}_frame_acc': _safe_div(correct, total)}
    if full:
        _add_class_metrics(metrics, class_counts)
    return metrics


def compute_lateral_decision_simple_metrics(prediction) -> Dict[str, float]:
    return _compute_decision_metrics(prediction, dimension='lateral', full=False)


def compute_lateral_decision_full_metrics(prediction) -> Dict[str, float]:
    return _compute_decision_metrics(prediction, dimension='lateral', full=True)


def compute_longitudinal_decision_simple_metrics(prediction) -> Dict[str, float]:
    return _compute_decision_metrics(prediction, dimension='longitudinal', full=False)


def compute_longitudinal_decision_full_metrics(prediction) -> Dict[str, float]:
    return _compute_decision_metrics(prediction, dimension='longitudinal', full=True)


def compute_lateral_decision_metrics(prediction) -> Dict[str, float]:
    return compute_lateral_decision_full_metrics(prediction)


def compute_longitudinal_decision_metrics(prediction) -> Dict[str, float]:
    return compute_longitudinal_decision_full_metrics(prediction)


metric_mapping['lateral_decision_simple'] = (compute_lateral_decision_simple_metrics, None)
metric_mapping['lateral_decision_full'] = (compute_lateral_decision_full_metrics, None)
metric_mapping['lateral_decision'] = (compute_lateral_decision_metrics, None)
metric_mapping['longitudinal_decision_simple'] = (compute_longitudinal_decision_simple_metrics, None)
metric_mapping['longitudinal_decision_full'] = (compute_longitudinal_decision_full_metrics, None)
metric_mapping['longitudinal_decision'] = (compute_longitudinal_decision_metrics, None)
