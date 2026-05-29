#!/usr/bin/env python3
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Add or remove eval markers in assistant labels for decision metrics.

The decision metric plugin recognizes marker lines like:

    __EVAL_SET__=city_a
    __EVAL_GROUP__=junction

Only the selected assistant message content is modified. By default, the script
marks the last assistant message in each sample, which is the label used by
ms-swift inference/evaluation. If --eval-group is omitted, no eval-group marker
is added.
"""

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional


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
LATERAL_DECISION_TOKENS = {
    'LAT_LANE_CHANGE_LEFT',
    'LAT_LANE_CHANGE_RIGHT',
    'LAT_NUDGE_LEFT',
    'LAT_NUDGE_RIGHT',
    'LAT_LANE_KEEP',
    'LAT_INTERSECTION_FOLLOW',
    'LAT_TURN_LEFT',
    'LAT_TURN_RIGHT',
    'LAT_U_TURN',
}
LONGITUDINAL_DECISION_NAME_TO_TOKEN = {
    '保持': 'LON_MAINTAIN',
    '加速': 'LON_ACCELERATE',
    '减速': 'LON_DECELERATE',
    '停车': 'LON_STOP',
}
LONGITUDINAL_DECISION_TOKENS = {
    'LON_MAINTAIN',
    'LON_ACCELERATE',
    'LON_DECELERATE',
    'LON_STOP',
}
EVAL_SET_MARKER_KEY = '__EVAL_SET__'
EVAL_GROUP_MARKER_KEY = '__EVAL_GROUP__'
MARKER_RE_TEMPLATE = r'^\s*{key}\s*=\s*[^\r\n]*(?:\r?\n)?'


def _default_output_path(input_path: Path, *, remove_markers: bool) -> Path:
    if remove_markers:
        return input_path.with_name(f'{input_path.stem}.without_eval_markers{input_path.suffix or ".jsonl"}')
    return input_path.with_name(f'{input_path.stem}.with_eval_set{input_path.suffix or ".jsonl"}')


def _read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON at {path}:{line_no}: {exc}') from exc


def _find_assistant_indices(messages: List[Dict], mode: str) -> List[int]:
    indices = [i for i, message in enumerate(messages) if message.get('role') == 'assistant']
    if mode == 'last':
        return indices[-1:] if indices else []
    if mode == 'all':
        return indices
    raise ValueError(f'Unsupported assistant mode: {mode}')


def _compile_marker_re(marker_key: str) -> re.Pattern:
    return re.compile(MARKER_RE_TEMPLATE.format(key=re.escape(marker_key)), re.MULTILINE)


def _strip_existing_marker(content: str, marker_key: str, *, count: int = 1) -> str:
    marker_re = _compile_marker_re(marker_key)
    return marker_re.sub('', content, count=count)


def _add_marker(content: str, marker_value: str, marker_key: str, replace_existing: bool) -> str:
    if not isinstance(content, str):
        raise TypeError(f'Assistant content must be a string, got {type(content)!r}.')
    if replace_existing:
        content = _strip_existing_marker(content, marker_key)
    marker = f'{marker_key}={marker_value}'
    if _compile_marker_re(marker_key).search(content):
        return content
    return f'{marker}\n{content}' if content else marker


def _sanitize_metric_component(value: str) -> str:
    value = value.strip()
    value = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in value)
    value = re.sub(r'_+', '_', value).strip('_')
    return value.lower() or 'unknown'


def _extract_tokens(content: str) -> List[str]:
    return [token.strip() for token in re.findall(r'<([^<>]+)>', content or '')]


def _normalize_lateral_decision(value: str) -> Optional[str]:
    value = value.strip().strip('<>').strip()
    if value in LATERAL_DECISION_TOKENS:
        return value
    return LATERAL_DECISION_NAME_TO_TOKEN.get(value)


def _normalize_longitudinal_decision(value: str) -> Optional[str]:
    value = value.strip().strip('<>').strip()
    if value in LONGITUDINAL_DECISION_TOKENS:
        return value
    return LONGITUDINAL_DECISION_NAME_TO_TOKEN.get(value)


def _extract_decision_group_from_content(content: str) -> str:
    lateral_decision = None
    longitudinal_decision = None
    for token in _extract_tokens(content):
        if lateral_decision is None:
            lateral_decision = _normalize_lateral_decision(token)
        if longitudinal_decision is None:
            longitudinal_decision = _normalize_longitudinal_decision(token)
        if lateral_decision is not None and longitudinal_decision is not None:
            return _sanitize_metric_component(f'{lateral_decision}_and_{longitudinal_decision}')
    raise ValueError(f'Cannot derive eval group from assistant content: {content!r}')


def add_eval_set_marker(row: Dict, eval_set: str, *, assistant_mode: str, marker_key: str,
                        replace_existing: bool, eval_group: Optional[str], eval_group_from_decision: bool,
                        group_marker_key: str) -> Dict:
    messages = row.get('messages')
    if not isinstance(messages, list):
        raise ValueError('Each row must contain a list field named "messages".')
    indices = _find_assistant_indices(messages, assistant_mode)
    if not indices:
        raise ValueError('No assistant message found in row.')
    for index in indices:
        message = messages[index]
        content = message.get('content', '')
        current_eval_group = eval_group
        if eval_group_from_decision:
            current_eval_group = _extract_decision_group_from_content(content)
        if current_eval_group:
            content = _add_marker(content, current_eval_group, group_marker_key, replace_existing=replace_existing)
        message['content'] = _add_marker(content, eval_set, marker_key, replace_existing=replace_existing)
    return row


def remove_eval_markers(row: Dict, *, assistant_mode: str, marker_key: str, group_marker_key: str) -> Dict:
    messages = row.get('messages')
    if not isinstance(messages, list):
        raise ValueError('Each row must contain a list field named "messages".')
    indices = _find_assistant_indices(messages, assistant_mode)
    if not indices:
        raise ValueError('No assistant message found in row.')
    for index in indices:
        message = messages[index]
        content = message.get('content', '')
        if not isinstance(content, str):
            raise TypeError(f'Assistant content must be a string, got {type(content)!r}.')
        content = _strip_existing_marker(content, marker_key, count=0)
        message['content'] = _strip_existing_marker(content, group_marker_key, count=0)
    return row


def _resolve_eval_set(input_path: Path, explicit_eval_set: Optional[str]) -> str:
    if explicit_eval_set:
        return explicit_eval_set
    return input_path.stem


def process_file(input_path: Path, output_path: Path, *, eval_set: str, assistant_mode: str, marker_key: str,
                 replace_existing: bool, eval_group: Optional[str], eval_group_from_decision: bool,
                 group_marker_key: str, remove_markers: bool) -> int:
    count = 0
    in_place = input_path.resolve() == output_path.resolve()
    if in_place:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = tempfile.NamedTemporaryFile(
            'w', encoding='utf-8', dir=output_path.parent, prefix=f'.{output_path.name}.', suffix='.tmp', delete=False)
        tmp_path = Path(tmp_file.name)
        f = tmp_file
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path
        f = output_path.open('w', encoding='utf-8')
    try:
        with f:
            for row in _read_jsonl(input_path):
                if remove_markers:
                    row = remove_eval_markers(
                        row, assistant_mode=assistant_mode, marker_key=marker_key, group_marker_key=group_marker_key)
                else:
                    row = add_eval_set_marker(
                        row,
                        eval_set,
                        assistant_mode=assistant_mode,
                        marker_key=marker_key,
                        replace_existing=replace_existing,
                        eval_group=eval_group,
                        eval_group_from_decision=eval_group_from_decision,
                        group_marker_key=group_marker_key)
                f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                count += 1
        if in_place:
            tmp_path.replace(output_path)
    except Exception:
        if in_place:
            tmp_path.unlink(missing_ok=True)
        raise
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Add or remove decision metric eval markers in assistant content in jsonl files.')
    parser.add_argument('inputs', nargs='+', type=Path, help='Input jsonl file(s).')
    parser.add_argument(
        '--eval-set',
        help='Eval-set name to write. Defaults to each input file stem.')
    parser.add_argument(
        '--eval-group',
        help='Optional eval-group name to write. If omitted, no eval-group marker is added.')
    parser.add_argument(
        '--eval-group-from-decision',
        action='store_true',
        help='Derive eval-group per sample from assistant decision tokens or Chinese decision names.')
    parser.add_argument(
        '--remove-markers',
        action='store_true',
        help='Remove eval-set and eval-group marker lines instead of adding markers.')
    parser.add_argument('--output', type=Path, help='Output jsonl path. Only valid for one input.')
    parser.add_argument('--output-dir', type=Path, help='Directory for output files when processing multiple inputs.')
    parser.add_argument(
        '--assistant-mode',
        choices=['last', 'all'],
        default='last',
        help='Which assistant message(s) to mark. Defaults to last.')
    parser.add_argument(
        '--marker-key',
        default=EVAL_SET_MARKER_KEY,
        help=f'Eval-set marker key. Defaults to {EVAL_SET_MARKER_KEY}.')
    parser.add_argument(
        '--group-marker-key',
        default=EVAL_GROUP_MARKER_KEY,
        help=f'Eval-group marker key. Defaults to {EVAL_GROUP_MARKER_KEY}.')
    parser.add_argument(
        '--replace-existing',
        action='store_true',
        help='Replace an existing marker line instead of leaving it unchanged.')
    parser.add_argument('--in-place', action='store_true', help='Modify input files in place.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output and len(args.inputs) != 1:
        raise ValueError('--output can only be used with exactly one input file.')
    if args.output and args.output_dir:
        raise ValueError('--output and --output-dir cannot be used together.')
    if args.in_place and (args.output or args.output_dir):
        raise ValueError('--in-place cannot be combined with --output or --output-dir.')
    if args.eval_group and args.eval_group_from_decision:
        raise ValueError('--eval-group and --eval-group-from-decision cannot be used together.')
    if args.remove_markers and args.eval_set:
        raise ValueError('--remove-markers cannot be combined with --eval-set.')
    if args.remove_markers and args.eval_group:
        raise ValueError('--remove-markers cannot be combined with --eval-group.')
    if args.remove_markers and args.eval_group_from_decision:
        raise ValueError('--remove-markers cannot be combined with --eval-group-from-decision.')

    for input_path in args.inputs:
        input_path = input_path.expanduser()
        if args.in_place:
            output_path = input_path
        elif args.output:
            output_path = args.output.expanduser()
        elif args.output_dir:
            output_path = args.output_dir.expanduser() / input_path.name
        else:
            output_path = _default_output_path(input_path, remove_markers=args.remove_markers)

        eval_set = '' if args.remove_markers else _resolve_eval_set(input_path, args.eval_set)
        count = process_file(
            input_path,
            output_path,
            eval_set=eval_set,
            assistant_mode=args.assistant_mode,
            marker_key=args.marker_key,
            replace_existing=args.replace_existing,
            eval_group=args.eval_group,
            eval_group_from_decision=args.eval_group_from_decision,
            group_marker_key=args.group_marker_key,
            remove_markers=args.remove_markers)
        eval_group_msg = ''
        if args.eval_group:
            eval_group_msg = f'  eval_group={args.eval_group}'
        elif args.eval_group_from_decision:
            eval_group_msg = '  eval_group=from_decision'
        if args.remove_markers:
            print(f'{input_path} -> {output_path}  mode=remove_markers  rows={count}')
        else:
            print(f'{input_path} -> {output_path}  eval_set={eval_set}{eval_group_msg}  rows={count}')


if __name__ == '__main__':
    main()
