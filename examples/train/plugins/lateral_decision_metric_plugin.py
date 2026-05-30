# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple

from swift.plugin.metric import metric_mapping
from swift.utils import Serializer


IGNORED_GENERATE_KWARGS = {
    'scene_id',
    'sample_token',
    'timestamp',
    'data_type_tag',
}


def _patch_generate_context_remove_unused_columns() -> None:
    from swift.llm.template.base import Template

    if getattr(Template, '_lateral_decision_generate_context_patched', False):
        return

    origin_generate_context = Template.generate_context

    @contextmanager
    def generate_context(self):
        remove_unused_columns = self.remove_unused_columns
        with origin_generate_context(self):
            self.remove_unused_columns = True
            try:
                yield
            finally:
                self.remove_unused_columns = remove_unused_columns

    Template.generate_context = generate_context
    Template._lateral_decision_generate_context_patched = True


def _patch_prepare_generate_kwargs() -> None:
    from swift.llm.template.base import Template

    if getattr(Template, '_lateral_decision_prepare_generate_kwargs_patched', False):
        return

    origin_prepare_generate_kwargs = Template.prepare_generate_kwargs

    def prepare_generate_kwargs(self, generate_kwargs, *, model=None):
        for key in IGNORED_GENERATE_KWARGS:
            generate_kwargs.pop(key, None)
        return origin_prepare_generate_kwargs(self, generate_kwargs, model=model)

    Template.prepare_generate_kwargs = prepare_generate_kwargs
    Template._lateral_decision_prepare_generate_kwargs_patched = True


_patch_generate_context_remove_unused_columns()
_patch_prepare_generate_kwargs()


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

EVAL_SET_MARKER_RE = re.compile(r'(?:^|\n)\s*__EVAL_SET__\s*=\s*([^\r\n]+)')
EVAL_GROUP_MARKER_RE = re.compile(r'(?:^|\n)\s*__EVAL_GROUP__\s*=\s*([^\r\n]+)')
UNKNOWN_EVAL_SET_NAME = 'unknown'


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


def _sanitize_metric_component(value: str) -> str:
    value = value.strip()
    value = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in value)
    value = re.sub(r'_+', '_', value).strip('_')
    return value.lower() or UNKNOWN_EVAL_SET_NAME


def extract_eval_set_from_labels(labels: str) -> Optional[str]:
    match = EVAL_SET_MARKER_RE.search(labels or '')
    if match is None:
        return None
    return _sanitize_metric_component(match.group(1))


def extract_eval_group_from_labels(labels: str) -> Optional[str]:
    match = EVAL_GROUP_MARKER_RE.search(labels or '')
    if match is None:
        return None
    return _sanitize_metric_component(match.group(1))


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


def _extract_answer_content(labels: str) -> str:
    answer_match = re.search(r'<answer>(.*?)</answer>', labels or '', re.DOTALL)
    return answer_match.group(1) if answer_match else labels


def extract_decision_gt_pairs_from_labels(labels: str) -> Set[Tuple[Optional[str], Optional[str]]]:
    answer_content = _extract_answer_content(labels)
    gt_pairs = set()
    for result in answer_content.split(';'):
        tokens = _extract_tokens(result)
        if not tokens:
            continue
        lateral = _normalize_lateral_decision(tokens[0])
        longitudinal = _normalize_longitudinal_decision(tokens[1] if len(tokens) >= 2 else None)
        if lateral is not None or longitudinal is not None:
            gt_pairs.add((lateral, longitudinal))
    return gt_pairs


def extract_decision_gt_from_labels(labels: str) -> Tuple[Set[str], Set[str]]:
    gt_pairs = extract_decision_gt_pairs_from_labels(labels)
    lateral_gt_set = {lateral for lateral, _ in gt_pairs if lateral is not None}
    longitudinal_gt_set = {longitudinal for _, longitudinal in gt_pairs if longitudinal is not None}
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


def _init_decision_stats(tokens: List[str]) -> Dict:
    return {
        'class_counts': _init_class_counts(tokens),
        'correct': 0,
        'total': 0,
    }


def _update_decision_stats(stats: Dict, pred: Optional[str], gt_set: Set[str]) -> None:
    stats['total'] += 1
    stats['correct'] += int(pred in gt_set)
    _update_class_counts(pred, gt_set, stats['class_counts'])


def _add_decision_stats_metrics(metrics: Dict[str, float], stats: Dict, metric_prefix: str, *, full: bool) -> None:
    metrics[f'{metric_prefix}_frame_acc'] = _safe_div(stats['correct'], stats['total'])
    if full:
        _add_class_metrics(metrics, stats['class_counts'])


def _add_prefixed_class_metrics(metrics: Dict[str, float], class_counts: Dict[str, Dict[str, int]],
                                prefix: str) -> None:
    class_metrics = {}
    _add_class_metrics(class_metrics, class_counts)
    for key, value in class_metrics.items():
        metrics[f'{prefix}_{key}'] = value


def _add_group_decision_stats_metrics(metrics: Dict[str, float], stats: Dict, metric_prefix: str, eval_set: str,
                                      *, full: bool) -> None:
    prefix = f'{metric_prefix}_{eval_set}'
    metrics[f'{prefix}_frame_acc'] = _safe_div(stats['correct'], stats['total'])
    if full:
        _add_prefixed_class_metrics(metrics, stats['class_counts'], prefix)


def _update_stats_by_key(stats_by_key: Dict[str, Dict], key: str, tokens: List[str], pred: Optional[str],
                         gt_set: Set[str]) -> None:
    if key not in stats_by_key:
        stats_by_key[key] = _init_decision_stats(tokens)
    _update_decision_stats(stats_by_key[key], pred, gt_set)


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

    overall_stats = _init_decision_stats(tokens)
    eval_set_stats: Dict[str, Dict] = {}
    eval_group_stats: Dict[str, Dict] = {}
    eval_set_group_stats: Dict[str, Dict] = {}
    has_eval_set_marker = False
    has_eval_group = False

    for i in range(preds.shape[0]):
        pred_text = _decode_serialized_text(preds[i])
        label_text = _decode_serialized_text(labels[i])
        eval_set = extract_eval_set_from_labels(label_text)
        eval_group = extract_eval_group_from_labels(label_text)
        has_eval_set_marker = has_eval_set_marker or eval_set is not None
        has_eval_group = has_eval_group or eval_group is not None

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

        _update_decision_stats(overall_stats, pred_decision, gt_set)
        eval_set = eval_set or UNKNOWN_EVAL_SET_NAME
        _update_stats_by_key(eval_set_stats, eval_set, tokens, pred_decision, gt_set)
        if eval_group is not None:
            _update_stats_by_key(eval_group_stats, f'group_{eval_group}', tokens, pred_decision, gt_set)
            _update_stats_by_key(eval_set_group_stats, f'{eval_set}_group_{eval_group}', tokens, pred_decision,
                                 gt_set)

    metrics = {}
    _add_decision_stats_metrics(metrics, overall_stats, metric_prefix, full=full)
    if has_eval_set_marker:
        for eval_set, stats in sorted(eval_set_stats.items()):
            _add_group_decision_stats_metrics(metrics, stats, metric_prefix, eval_set, full=full)
    if has_eval_group:
        for eval_group, stats in sorted(eval_group_stats.items()):
            _add_group_decision_stats_metrics(metrics, stats, metric_prefix, eval_group, full=full)
        if has_eval_set_marker:
            for eval_set_group, stats in sorted(eval_set_group_stats.items()):
                _add_group_decision_stats_metrics(metrics, stats, metric_prefix, eval_set_group, full=full)
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
