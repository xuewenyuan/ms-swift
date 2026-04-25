# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from typing import Dict, List, Optional, Set

from swift.plugin.metric import metric_mapping
from swift.utils import Serializer


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


def _extract_tokens(text: str) -> List[str]:
    return [token.strip() for token in re.findall(r'<([^<>]+)>', text or '')]


def extract_lateral_from_response(response: str) -> Optional[str]:
    if '</think>' in response:
        response = response.split('</think>', 1)[1]

    for token in _extract_tokens(response):
        lateral = _normalize_lateral_decision(token)
        if lateral is not None:
            return lateral
    return None


def extract_lateral_gt_from_labels(labels: str) -> Set[str]:
    answer_match = re.search(r'<answer>(.*?)</answer>', labels or '', re.DOTALL)
    answer_content = answer_match.group(1) if answer_match else labels

    gt_set = set()
    for result in answer_content.split(';'):
        tokens = _extract_tokens(result)
        if not tokens:
            continue
        lateral = _normalize_lateral_decision(tokens[0])
        if lateral is not None:
            gt_set.add(lateral)
    return gt_set


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_lateral_decision_metrics(prediction) -> Dict[str, float]:
    preds, labels = prediction[0], prediction[1]
    class_counts = {
        token: {
            'tp': 0,
            'fp': 0,
            'fn': 0
        }
        for token in LATERAL_DECISION_TOKENS
    }
    total = 0
    correct = 0

    for i in range(preds.shape[0]):
        pred_text = _decode_serialized_text(preds[i])
        label_text = _decode_serialized_text(labels[i])

        pred_lateral = extract_lateral_from_response(pred_text)
        gt_lateral_set = extract_lateral_gt_from_labels(label_text)
        if not gt_lateral_set:
            continue

        total += 1
        is_correct = pred_lateral in gt_lateral_set
        correct += int(is_correct)

        for token in LATERAL_DECISION_TOKENS:
            if pred_lateral == token and token in gt_lateral_set:
                class_counts[token]['tp'] += 1
            elif pred_lateral == token and token not in gt_lateral_set:
                class_counts[token]['fp'] += 1
            elif pred_lateral != token and token in gt_lateral_set:
                class_counts[token]['fn'] += 1

    metrics = {'lateral_frame_acc': _safe_div(correct, total)}
    for token, counts in class_counts.items():
        prefix = token.lower()
        precision = _safe_div(counts['tp'], counts['tp'] + counts['fp'])
        recall = _safe_div(counts['tp'], counts['tp'] + counts['fn'])
        f1 = _safe_div(2 * precision * recall, precision + recall)
        metrics[f'{prefix}_precision'] = precision
        metrics[f'{prefix}_recall'] = recall
        metrics[f'{prefix}_f1'] = f1
    return metrics


metric_mapping['lateral_decision'] = (compute_lateral_decision_metrics, None)
