#!/usr/bin/env python3
"""Filter JSONL samples by required auxiliary label keys.

The script checks each JSONL record directly and, when a label path field is
present, also loads the referenced label object using the same common formats as
the Qwen3-VL aux label loader. Kept records are written unchanged.
"""

import argparse
import json
import os
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import msgpack
    import msgpack_numpy as msgpack_numpy
except ImportError:  # pragma: no cover - optional dependency
    msgpack = None
    msgpack_numpy = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional dependency
    zstd = None

try:
    import moxing as mox
except ImportError:  # pragma: no cover - optional dependency
    mox = None


# =========================
# Configuration
# =========================
# Keys are checked against the JSONL record first. If the record contains a
# label path field, the loaded label object is also checked. Dot paths are
# supported for nested dictionaries/lists.
DEFAULT_LABEL_PATH_KEYS = ('labels_path', 'aux_label_path', 'label_path', 'src_label_path', 'labels_file')
DEFAULT_LABEL_FORMAT = 'auto'
DEFAULT_SRC_LABEL_PATH = '2,0'
DEFAULT_MAX_EXAMPLES = 10

# BevAuxLoss.forward currently requires these fields when bev_distill=True.
# bevseg_cache_valid and bev_feat_cache_valid are optional masks.
BEV_LOSS_REQUIRED_KEYS = (
    'centerpoint_head_obj_white',
    'rl_sample_mask',
    'is_cls_only',
    'pnc_bev_input.dense_bev_feat',
)

# GodAuxLoss.forward currently requires these fields when default loss_tasks
# are enabled and god_distill=True. godseg_cache_valid and god_feat_cache_valid
# are optional masks.
GOD_LOSS_REQUIRED_KEYS = (
    'pnc_god_head_input.occupancy',
    'pnc_god_head_input.semantic',
    'pnc_god_head_input.visibility',
    'pnc_god_head_input.drivable',
    'pnc_god_dense_input',
)

DEFAULT_REQUIRED_KEYS = BEV_LOSS_REQUIRED_KEYS + GOD_LOSS_REQUIRED_KEYS
PRESET_KEYS = {
    'bev': BEV_LOSS_REQUIRED_KEYS,
    'bev_distill': BEV_LOSS_REQUIRED_KEYS,
    'god': GOD_LOSS_REQUIRED_KEYS,
    'god_distill': GOD_LOSS_REQUIRED_KEYS,
}
PRESET_KEYS['bev_god'] = DEFAULT_REQUIRED_KEYS
PRESET_KEYS['bev_god_distill'] = DEFAULT_REQUIRED_KEYS


def is_obs_path(path: str) -> bool:
    return path.startswith('obs://') or path.startswith('obs:')


def read_binary(path: str) -> bytes:
    if is_obs_path(path):
        if mox is None:
            raise ImportError('moxing is required to read OBS paths.')
        return mox.file.read(path, binary=True)
    with open(path, 'rb') as f:
        return f.read()


def parse_path_spec(raw: str) -> List[Any]:
    if raw == '':
        return []
    path = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            path.append(int(item))
        except ValueError:
            path.append(item)
    return path


def resolve_data_path(data: Any, path_spec: Sequence[Any]) -> Any:
    current = data
    for key in path_spec:
        current = current[key]
    return current


def infer_label_format(path: str, label_format: str) -> str:
    if label_format != 'auto':
        return label_format
    if path.endswith(('.zstd.mspack', '.mspack')):
        return 'mspack'
    if path.endswith('.json'):
        return 'json'
    if path.endswith('.npy'):
        return 'npy'
    if path.endswith('.npz'):
        return 'npz'
    return 'mspack'


def load_label(path: str, label_format: str, src_label_path: Sequence[Any]) -> Any:
    resolved_format = infer_label_format(path, label_format)
    payload = read_binary(path)
    if resolved_format == 'mspack':
        if msgpack is None or msgpack_numpy is None:
            raise ImportError('msgpack and msgpack_numpy are required to read mspack labels.')
        if zstd is not None:
            try:
                payload = zstd.ZstdDecompressor().decompress(payload)
            except Exception:
                pass
        data = msgpack.unpackb(payload, object_hook=msgpack_numpy.decode, raw=False, strict_map_key=False)
    elif resolved_format == 'json':
        data = json.loads(payload.decode('utf-8'))
    elif resolved_format == 'npy':
        if np is None:
            raise ImportError('numpy is required to read npy labels.')
        data = np.load(BytesIO(payload), allow_pickle=True)
        if hasattr(data, 'item') and data.shape == ():
            data = data.item()
    elif resolved_format == 'npz':
        if np is None:
            raise ImportError('numpy is required to read npz labels.')
        data = dict(np.load(BytesIO(payload), allow_pickle=True))
    else:
        raise ValueError(f'Unsupported label format: {resolved_format}')
    return resolve_data_path(data, src_label_path)


def iter_path_values(record: Mapping[str, Any], label_path_keys: Sequence[str]) -> Iterable[Tuple[str, str]]:
    for key in label_path_keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            yield key, value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    yield key, item


def split_required_key(raw: str) -> List[Any]:
    path = []
    for item in raw.split('.'):
        if item == '':
            continue
        try:
            path.append(int(item))
        except ValueError:
            path.append(item)
    return path


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return len(value) == 0
    if np is not None and isinstance(value, np.ndarray):
        return value.size == 0
    return False


def has_required_path(source: Any, raw_key: str, allow_empty: bool) -> bool:
    current = source
    for key in split_required_key(raw_key):
        if isinstance(current, Mapping):
            if key not in current:
                return False
            current = current[key]
        elif isinstance(current, (list, tuple)) and isinstance(key, int):
            if key < 0 or key >= len(current):
                return False
            current = current[key]
        else:
            return False
    return allow_empty or not is_empty_value(current)


def collect_required_keys(args: argparse.Namespace) -> List[str]:
    keys: List[str] = []
    for preset in args.preset or []:
        keys.extend(PRESET_KEYS[preset])
    for raw in args.required_key or []:
        keys.extend(item.strip() for item in raw.split(',') if item.strip())
    if not keys:
        keys.extend(DEFAULT_REQUIRED_KEYS)
    seen = set()
    deduped = []
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def default_output_path(input_path: str) -> str:
    path = Path(input_path)
    if path.suffix == '.jsonl':
        return str(path.with_name(f'{path.stem}.filtered.jsonl'))
    return f'{input_path}.filtered.jsonl'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Filter JSONL samples missing required auxiliary label keys.')
    parser.add_argument('jsonl', help='Input JSONL file.')
    parser.add_argument('-o', '--output', help='Output JSONL file. Defaults to <input>.filtered.jsonl.')
    parser.add_argument(
        '--required-key',
        action='append',
        help='Required key or dot path. Can be repeated or comma-separated. Defaults to DEFAULT_REQUIRED_KEYS.',
    )
    parser.add_argument(
        '--preset',
        action='append',
        choices=sorted(PRESET_KEYS),
        help='Add a built-in required-key set, e.g. bev, god, bev_god, bev_god_distill.',
    )
    parser.add_argument(
        '--label-path-key',
        action='append',
        help='JSONL field containing external label paths. Defaults to common aux label path keys.',
    )
    parser.add_argument('--label-format', default=DEFAULT_LABEL_FORMAT, choices=('auto', 'mspack', 'json', 'npy', 'npz'))
    parser.add_argument(
        '--src-label-path',
        default=DEFAULT_SRC_LABEL_PATH,
        help='Comma-separated path inside loaded label object. Use empty string to disable. Default: 2,0.',
    )
    parser.add_argument('--allow-empty', action='store_true', help='Treat present-but-empty values as valid.')
    parser.add_argument('--dry-run', action='store_true', help='Only print statistics; do not write output.')
    parser.add_argument('--max-examples', type=int, default=DEFAULT_MAX_EXAMPLES, help='Max dropped examples to print.')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.jsonl
    output_path = args.output or default_output_path(input_path)
    if not args.dry_run and os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError('Output path must be different from input path.')

    required_keys = collect_required_keys(args)
    label_path_keys = tuple(args.label_path_key or DEFAULT_LABEL_PATH_KEYS)
    src_label_path = parse_path_spec(args.src_label_path)
    label_cache: Dict[str, Any] = {}

    total = kept = dropped = blank = invalid_json = label_load_errors = 0
    missing_counter: Counter[str] = Counter()
    dropped_examples = []

    out_f = None if args.dry_run else open(output_path, 'w', encoding='utf-8')
    try:
        with open(input_path, 'r', encoding='utf-8') as in_f:
            for line_no, line in enumerate(in_f, 1):
                raw_line = line.rstrip('\n')
                if not raw_line.strip():
                    blank += 1
                    continue
                total += 1
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    invalid_json += 1
                    dropped += 1
                    if len(dropped_examples) < args.max_examples:
                        dropped_examples.append((line_no, ['<invalid_json>'], str(exc), None))
                    continue
                if not isinstance(record, dict):
                    dropped += 1
                    missing_counter['<record_not_object>'] += 1
                    if len(dropped_examples) < args.max_examples:
                        dropped_examples.append((line_no, ['<record_not_object>'], None, None))
                    continue

                sources = [record]
                label_error: Optional[str] = None
                first_label_path: Optional[str] = None
                for _, label_path in iter_path_values(record, label_path_keys):
                    first_label_path = first_label_path or label_path
                    try:
                        if label_path not in label_cache:
                            label_cache[label_path] = load_label(label_path, args.label_format, src_label_path)
                        sources.append(label_cache[label_path])
                    except Exception as exc:
                        label_load_errors += 1
                        label_error = f'{type(exc).__name__}: {exc}'

                missing = [
                    key for key in required_keys
                    if not any(has_required_path(source, key, args.allow_empty) for source in sources)
                ]
                if missing:
                    dropped += 1
                    missing_counter.update(missing)
                    if label_error:
                        missing_counter['<label_load_error>'] += 1
                    if len(dropped_examples) < args.max_examples:
                        dropped_examples.append((line_no, missing, label_error, first_label_path))
                    continue

                kept += 1
                if out_f is not None:
                    out_f.write(raw_line + '\n')
    finally:
        if out_f is not None:
            out_f.close()

    print(f'input: {input_path}')
    print(f'output: {output_path if not args.dry_run else "<dry-run>"}')
    print(f'required_keys: {", ".join(required_keys)}')
    print(f'label_path_keys: {", ".join(label_path_keys)}')
    print(f'src_label_path: {src_label_path}')
    print(f'total_nonblank_lines: {total}')
    print(f'kept: {kept}')
    print(f'dropped: {dropped}')
    print(f'blank_lines_skipped: {blank}')
    print(f'invalid_json: {invalid_json}')
    print(f'label_load_errors: {label_load_errors}')
    if total:
        print(f'kept_ratio: {kept / total:.6f}')
        print(f'dropped_ratio: {dropped / total:.6f}')
    if missing_counter:
        print('drop_reasons:')
        for key, count in missing_counter.most_common():
            print(f'  {key}: {count}')
    if dropped_examples:
        print('dropped_examples:')
        for line_no, missing, label_error, label_path in dropped_examples:
            print(f'  line {line_no}: missing={missing}, label_path={label_path}, label_error={label_error}')


if __name__ == '__main__':
    main()
