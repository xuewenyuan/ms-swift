#!/usr/bin/env python3
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Add eval markers to assistant labels for decision metrics.

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


EVAL_SET_MARKER_KEY = '__EVAL_SET__'
EVAL_GROUP_MARKER_KEY = '__EVAL_GROUP__'
MARKER_RE_TEMPLATE = r'(?:^|\n)\s*{key}\s*=\s*[^\r\n]*(?:\r?\n)?'


def _default_output_path(input_path: Path) -> Path:
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


def _strip_existing_marker(content: str, marker_key: str) -> str:
    marker_re = re.compile(MARKER_RE_TEMPLATE.format(key=re.escape(marker_key)))
    return marker_re.sub('', content, count=1)


def _add_marker(content: str, marker_value: str, marker_key: str, replace_existing: bool) -> str:
    if not isinstance(content, str):
        raise TypeError(f'Assistant content must be a string, got {type(content)!r}.')
    if replace_existing:
        content = _strip_existing_marker(content, marker_key)
    marker = f'{marker_key}={marker_value}'
    if re.search(MARKER_RE_TEMPLATE.format(key=re.escape(marker_key)), content):
        return content
    return f'{marker}\n{content}' if content else marker


def add_eval_set_marker(row: Dict, eval_set: str, *, assistant_mode: str, marker_key: str,
                        replace_existing: bool, eval_group: Optional[str], group_marker_key: str) -> Dict:
    messages = row.get('messages')
    if not isinstance(messages, list):
        raise ValueError('Each row must contain a list field named "messages".')
    indices = _find_assistant_indices(messages, assistant_mode)
    if not indices:
        raise ValueError('No assistant message found in row.')
    for index in indices:
        message = messages[index]
        content = message.get('content', '')
        if eval_group:
            content = _add_marker(content, eval_group, group_marker_key, replace_existing=replace_existing)
        message['content'] = _add_marker(content, eval_set, marker_key, replace_existing=replace_existing)
    return row


def _resolve_eval_set(input_path: Path, explicit_eval_set: Optional[str]) -> str:
    if explicit_eval_set:
        return explicit_eval_set
    return input_path.stem


def process_file(input_path: Path, output_path: Path, *, eval_set: str, assistant_mode: str, marker_key: str,
                 replace_existing: bool, eval_group: Optional[str], group_marker_key: str) -> int:
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
                row = add_eval_set_marker(
                    row,
                    eval_set,
                    assistant_mode=assistant_mode,
                    marker_key=marker_key,
                    replace_existing=replace_existing,
                    eval_group=eval_group,
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
        description='Add decision metric eval markers to assistant content in jsonl files.')
    parser.add_argument('inputs', nargs='+', type=Path, help='Input jsonl file(s).')
    parser.add_argument(
        '--eval-set',
        help='Eval-set name to write. Defaults to each input file stem.')
    parser.add_argument(
        '--eval-group',
        help='Optional eval-group name to write. If omitted, no eval-group marker is added.')
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

    for input_path in args.inputs:
        input_path = input_path.expanduser()
        if args.in_place:
            output_path = input_path
        elif args.output:
            output_path = args.output.expanduser()
        elif args.output_dir:
            output_path = args.output_dir.expanduser() / input_path.name
        else:
            output_path = _default_output_path(input_path)

        eval_set = _resolve_eval_set(input_path, args.eval_set)
        count = process_file(
            input_path,
            output_path,
            eval_set=eval_set,
            assistant_mode=args.assistant_mode,
            marker_key=args.marker_key,
            replace_existing=args.replace_existing,
            eval_group=args.eval_group,
            group_marker_key=args.group_marker_key)
        eval_group_msg = f'  eval_group={args.eval_group}' if args.eval_group else ''
        print(f'{input_path} -> {output_path}  eval_set={eval_set}{eval_group_msg}  rows={count}')


if __name__ == '__main__':
    main()
