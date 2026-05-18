#!/usr/bin/env python3
"""Filter JSONL samples by required auxiliary label keys.

The script checks each JSONL record directly and, when a label path field is
present, also loads the referenced label object using the same common formats as
the Qwen3-VL aux label loader. Kept records are written unchanged.
"""

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
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
DEFAULT_NUM_WORKERS = max(1, min(8, os.cpu_count() or 1))
DEFAULT_CHUNK_SIZE = 512
DEFAULT_PROGRESS_MIN_INTERVAL_BYTES = 8 * 1024 * 1024

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


class ProgressReporter:

    def __init__(self, *, total_bytes: int, enabled: bool):
        self.total_bytes = max(0, total_bytes)
        self.enabled = enabled and self.total_bytes > 0
        self.processed_bytes = 0
        self._bar = None
        self._last_reported_bytes = 0
        if not self.enabled:
            return
        try:
            from tqdm import tqdm
            self._bar = tqdm(total=self.total_bytes, unit='B', unit_scale=True, desc='filter', dynamic_ncols=True)
        except Exception:
            self._bar = None
            self._print_progress(force=True)

    def update(self, amount: int) -> None:
        if not self.enabled or amount <= 0:
            return
        self.processed_bytes += amount
        if self._bar is not None:
            self._bar.update(amount)
            return
        threshold = max(self.total_bytes // 100, DEFAULT_PROGRESS_MIN_INTERVAL_BYTES)
        if self.processed_bytes - self._last_reported_bytes >= threshold or self.processed_bytes >= self.total_bytes:
            self._print_progress(force=self.processed_bytes >= self.total_bytes)

    def close(self) -> None:
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.close()
        else:
            self._print_progress(force=True)
            sys.stderr.write('\n')
            sys.stderr.flush()

    def _print_progress(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        processed = min(self.processed_bytes, self.total_bytes)
        if not force and processed == self._last_reported_bytes:
            return
        percent = 100.0 * processed / self.total_bytes if self.total_bytes else 100.0
        sys.stderr.write(f'\rfilter: {processed}/{self.total_bytes} bytes ({percent:.2f}%)')
        sys.stderr.flush()
        self._last_reported_bytes = processed


def new_stats() -> Dict[str, int]:
    return {
        'total': 0,
        'kept': 0,
        'dropped': 0,
        'blank': 0,
        'invalid_json': 0,
        'label_load_errors': 0,
    }


def process_chunk(chunk: List[Tuple[int, str]], settings: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = settings['required_keys']
    label_path_keys = settings['label_path_keys']
    label_format = settings['label_format']
    src_label_path = settings['src_label_path']
    allow_empty = settings['allow_empty']
    max_examples = settings['max_examples']

    label_cache: Dict[str, Any] = {}
    stats = new_stats()
    missing_counter: Counter = Counter()
    dropped_examples = []
    kept_lines = []

    for line_no, raw_line in chunk:
        if not raw_line.strip():
            stats['blank'] += 1
            continue
        stats['total'] += 1
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            stats['invalid_json'] += 1
            stats['dropped'] += 1
            if len(dropped_examples) < max_examples:
                dropped_examples.append((line_no, ['<invalid_json>'], str(exc), None))
            continue
        if not isinstance(record, dict):
            stats['dropped'] += 1
            missing_counter['<record_not_object>'] += 1
            if len(dropped_examples) < max_examples:
                dropped_examples.append((line_no, ['<record_not_object>'], None, None))
            continue

        sources = [record]
        label_error: Optional[str] = None
        first_label_path: Optional[str] = None
        for _, label_path in iter_path_values(record, label_path_keys):
            first_label_path = first_label_path or label_path
            try:
                if label_path not in label_cache:
                    label_cache[label_path] = load_label(label_path, label_format, src_label_path)
                sources.append(label_cache[label_path])
            except Exception as exc:
                stats['label_load_errors'] += 1
                label_error = f'{type(exc).__name__}: {exc}'

        missing = [
            key for key in required_keys if not any(has_required_path(source, key, allow_empty) for source in sources)
        ]
        if missing:
            stats['dropped'] += 1
            missing_counter.update(missing)
            if label_error:
                missing_counter['<label_load_error>'] += 1
            if len(dropped_examples) < max_examples:
                dropped_examples.append((line_no, missing, label_error, first_label_path))
            continue

        stats['kept'] += 1
        kept_lines.append(raw_line)

    return {
        'stats': stats,
        'missing_counter': dict(missing_counter),
        'dropped_examples': dropped_examples,
        'kept_lines': kept_lines,
    }


def iter_chunks(input_path: str, chunk_size: int) -> Iterable[Tuple[List[Tuple[int, str]], int]]:
    chunk: List[Tuple[int, str]] = []
    chunk_bytes = 0
    with open(input_path, 'r', encoding='utf-8') as in_f:
        for line_no, line in enumerate(in_f, 1):
            chunk_bytes += len(line.encode('utf-8'))
            chunk.append((line_no, line.rstrip('\n')))
            if len(chunk) >= chunk_size:
                yield chunk, chunk_bytes
                chunk = []
                chunk_bytes = 0
    if chunk:
        yield chunk, chunk_bytes


def merge_result(
    result: Dict[str, Any],
    stats: Dict[str, int],
    missing_counter: Counter,
    dropped_examples: List[Tuple[int, List[str], Optional[str], Optional[str]]],
    max_examples: int,
) -> None:
    for key, value in result['stats'].items():
        stats[key] += value
    missing_counter.update(result['missing_counter'])
    remaining = max(0, max_examples - len(dropped_examples))
    if remaining:
        dropped_examples.extend(result['dropped_examples'][:remaining])


def write_kept_lines(out_f: Optional[Any], kept_lines: Sequence[str]) -> None:
    if out_f is None:
        return
    for raw_line in kept_lines:
        out_f.write(raw_line + '\n')


def run_sequential(
    input_path: str,
    out_f: Optional[Any],
    settings: Dict[str, Any],
    chunk_size: int,
    progress: ProgressReporter,
) -> Dict[str, Any]:
    stats = new_stats()
    missing_counter: Counter = Counter()
    dropped_examples = []
    for chunk, chunk_bytes in iter_chunks(input_path, chunk_size):
        result = process_chunk(chunk, settings)
        merge_result(result, stats, missing_counter, dropped_examples, settings['max_examples'])
        write_kept_lines(out_f, result['kept_lines'])
        progress.update(chunk_bytes)
    return {
        'stats': stats,
        'missing_counter': missing_counter,
        'dropped_examples': dropped_examples,
    }


def run_parallel(
    input_path: str,
    out_f: Optional[Any],
    settings: Dict[str, Any],
    *,
    chunk_size: int,
    num_workers: int,
    unordered: bool,
    backend: str,
    progress: ProgressReporter,
) -> Dict[str, Any]:
    stats = new_stats()
    missing_counter: Counter = Counter()
    dropped_examples = []
    max_pending = max(num_workers * 4, num_workers)
    pending = {}
    ready: Dict[int, Dict[str, Any]] = {}
    next_to_write = 0
    next_index = 0
    chunks = iter_chunks(input_path, chunk_size)

    def submit_available(executor: ProcessPoolExecutor) -> None:
        nonlocal next_index
        while len(pending) < max_pending:
            try:
                chunk, chunk_bytes = next(chunks)
            except StopIteration:
                return
            future = executor.submit(process_chunk, chunk, settings)
            pending[future] = (next_index, chunk_bytes)
            next_index += 1

    executor_cls = ProcessPoolExecutor if backend == 'process' else ThreadPoolExecutor
    with executor_cls(max_workers=num_workers) as executor:
        submit_available(executor)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index, chunk_bytes = pending.pop(future)
                result = future.result()
                if unordered:
                    merge_result(result, stats, missing_counter, dropped_examples, settings['max_examples'])
                    write_kept_lines(out_f, result['kept_lines'])
                    progress.update(chunk_bytes)
                else:
                    ready[index] = {'result': result, 'chunk_bytes': chunk_bytes}
            if not unordered:
                while next_to_write in ready:
                    ready_item = ready.pop(next_to_write)
                    result = ready_item['result']
                    merge_result(result, stats, missing_counter, dropped_examples, settings['max_examples'])
                    write_kept_lines(out_f, result['kept_lines'])
                    progress.update(ready_item['chunk_bytes'])
                    next_to_write += 1
            submit_available(executor)

    return {
        'stats': stats,
        'missing_counter': missing_counter,
        'dropped_examples': dropped_examples,
    }


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
    parser.add_argument(
        '--num-workers',
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help='Number of parallel workers. Use 1 to disable parallelism.',
    )
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE, help='JSONL lines per worker task.')
    parser.add_argument(
        '--parallel-backend',
        choices=('process', 'thread'),
        default='process',
        help='Parallel backend. process is better for CPU-heavy decoding; thread is useful for I/O-heavy label loads.',
    )
    parser.add_argument(
        '--unordered',
        action='store_true',
        help='Write kept rows as soon as chunks finish. Faster, but output order may differ from input.',
    )
    parser.add_argument('--no-progress', action='store_true', help='Disable overall progress display.')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.jsonl
    output_path = args.output or default_output_path(input_path)
    if not args.dry_run and os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError('Output path must be different from input path.')
    if args.num_workers < 1:
        raise ValueError('--num-workers must be >= 1.')
    if args.chunk_size < 1:
        raise ValueError('--chunk-size must be >= 1.')

    required_keys = collect_required_keys(args)
    label_path_keys = tuple(args.label_path_key or DEFAULT_LABEL_PATH_KEYS)
    src_label_path = parse_path_spec(args.src_label_path)
    settings = {
        'required_keys': required_keys,
        'label_path_keys': label_path_keys,
        'label_format': args.label_format,
        'src_label_path': src_label_path,
        'allow_empty': args.allow_empty,
        'max_examples': args.max_examples,
    }

    out_f = None if args.dry_run else open(output_path, 'w', encoding='utf-8')
    parallel_backend = args.parallel_backend
    total_bytes = os.path.getsize(input_path)
    progress = ProgressReporter(total_bytes=total_bytes, enabled=not args.no_progress)
    try:
        if args.num_workers == 1:
            result = run_sequential(input_path, out_f, settings, args.chunk_size, progress)
        else:
            try:
                result = run_parallel(
                    input_path,
                    out_f,
                    settings,
                    chunk_size=args.chunk_size,
                    num_workers=args.num_workers,
                    unordered=args.unordered,
                    backend=parallel_backend,
                    progress=progress,
                )
            except PermissionError:
                if parallel_backend != 'process':
                    raise
                parallel_backend = 'thread'
                print('warning: process backend is unavailable; falling back to thread backend.')
                result = run_parallel(
                    input_path,
                    out_f,
                    settings,
                    chunk_size=args.chunk_size,
                    num_workers=args.num_workers,
                    unordered=args.unordered,
                    backend=parallel_backend,
                    progress=progress,
                )
    finally:
        progress.close()
        if out_f is not None:
            out_f.close()

    stats = result['stats']
    missing_counter = result['missing_counter']
    dropped_examples = result['dropped_examples']
    total = stats['total']

    print(f'input: {input_path}')
    print(f'output: {output_path if not args.dry_run else "<dry-run>"}')
    print(f'num_workers: {args.num_workers}')
    print(f'chunk_size: {args.chunk_size}')
    print(f'parallel_backend: {parallel_backend if args.num_workers > 1 else "none"}')
    print(f'unordered: {args.unordered}')
    print(f'progress: {not args.no_progress}')
    print(f'required_keys: {", ".join(required_keys)}')
    print(f'label_path_keys: {", ".join(label_path_keys)}')
    print(f'src_label_path: {src_label_path}')
    print(f'total_nonblank_lines: {stats["total"]}')
    print(f'kept: {stats["kept"]}')
    print(f'dropped: {stats["dropped"]}')
    print(f'blank_lines_skipped: {stats["blank"]}')
    print(f'invalid_json: {stats["invalid_json"]}')
    print(f'label_load_errors: {stats["label_load_errors"]}')
    if total:
        print(f'kept_ratio: {stats["kept"] / total:.6f}')
        print(f'dropped_ratio: {stats["dropped"] / total:.6f}')
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
