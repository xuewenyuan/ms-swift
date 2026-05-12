from collections import OrderedDict
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence

import json

import numpy as np

try:
    import msgpack
    import msgpack_numpy as msgpack_numpy
except ImportError:  # pragma: no cover - optional dependency
    msgpack = None
    msgpack_numpy = None

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional dependency
    zstd = None

try:
    import moxing as mox
except ImportError:  # pragma: no cover - optional dependency
    mox = None


def _is_path_like(value: Any) -> bool:
    return isinstance(value, str) and (value.startswith('obs://') or value.startswith('obs:') or '/' in value
                                       or value.endswith(('.npy', '.npz', '.json', '.mspack', '.zstd.mspack')))


def _is_obs_path(path: str) -> bool:
    return path.startswith('obs://') or path.startswith('obs:')


def _read_binary(path: str) -> bytes:
    if _is_obs_path(path):
        if mox is None:
            raise ImportError('moxing is required to read OBS label paths.')
        return mox.file.read(path, binary=True)
    with open(path, 'rb') as f:
        return f.read()


def _resolve_data_path(data: Any, path_spec: Sequence[Any]) -> Any:
    current = data
    for key in path_spec:
        current = current[key]
    return current


class Qwen3VLAuxLabelLoader:

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        label_loader_cfg = cfg['shared'].get('label_loader', {})
        self.path_key = label_loader_cfg.get('path_key', 'labels_path')
        self.label_format = label_loader_cfg.get('format', 'mspack')
        self.src_label_path = label_loader_cfg.get('src_label_path', [2, 0])
        self.cache_size = int(label_loader_cfg.get('cache_size', 256))
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._zstd = zstd.ZstdDecompressor() if zstd is not None else None

    def _cache_get(self, path: str) -> Any:
        if path not in self._cache:
            return None
        value = self._cache.pop(path)
        self._cache[path] = value
        return value

    def _cache_put(self, path: str, value: Any) -> Any:
        if self.cache_size <= 0:
            return value
        self._cache[path] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def _load_mspack(self, path: str) -> Any:
        if msgpack is None or msgpack_numpy is None:
            raise ImportError('msgpack and msgpack_numpy are required to load mspack labels.')
        payload = _read_binary(path)
        if self._zstd is not None:
            try:
                payload = self._zstd.decompress(payload)
            except Exception:
                pass
        return msgpack.unpackb(
            payload,
            object_hook=msgpack_numpy.decode,
            raw=False,
            strict_map_key=False,
        )

    def _load_json(self, path: str) -> Any:
        if _is_obs_path(path):
            return json.loads(_read_binary(path).decode('utf-8'))
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_generic(self, path: str) -> Any:
        label_format = self.label_format
        if label_format == 'auto':
            if path.endswith(('.zstd.mspack', '.mspack')):
                label_format = 'mspack'
            elif path.endswith('.npy'):
                label_format = 'npy'
            elif path.endswith('.npz'):
                label_format = 'npz'
            elif path.endswith('.json'):
                label_format = 'json'
            else:
                label_format = 'mspack'

        if label_format == 'mspack':
            data = self._load_mspack(path)
            return _resolve_data_path(data, self.src_label_path)
        if label_format == 'npy':
            if _is_obs_path(path):
                return np.load(BytesIO(_read_binary(path)), allow_pickle=True)
            return np.load(path, allow_pickle=True)
        if label_format == 'npz':
            if _is_obs_path(path):
                return dict(np.load(BytesIO(_read_binary(path)), allow_pickle=True))
            return dict(np.load(path, allow_pickle=True))
        if label_format == 'json':
            return self._load_json(path)
        raise ValueError(f'Unsupported label format: {label_format}')

    def load_path(self, path: str) -> Any:
        cached = self._cache_get(path)
        if cached is not None:
            return cached
        value = self._load_generic(path)
        return self._cache_put(path, value)

    @staticmethod
    def _normalize_batch_value(value: Any, batch_size: int) -> List[Any]:
        if value is None:
            return [None] * batch_size
        if isinstance(value, list):
            if len(value) == batch_size:
                return value
            if batch_size == 1:
                return [value]
            return value + [None] * max(0, batch_size - len(value))
        if isinstance(value, tuple):
            value = list(value)
            if len(value) == batch_size:
                return value
            if batch_size == 1:
                return [tuple(value)]
        return [value] if batch_size == 1 else [value] + [None] * (batch_size - 1)

    def _resolve_shared_labels(self, batch_kwargs: Dict[str, Any], batch_size: int) -> List[Any]:
        paths = self._normalize_batch_value(batch_kwargs.get(self.path_key), batch_size)
        resolved = []
        for path in paths:
            if _is_path_like(path):
                resolved.append(self.load_path(path))
            else:
                resolved.append(None)
        return resolved

    def _resolve_task_target(self, value: Any, batch_size: int) -> List[Any]:
        values = self._normalize_batch_value(value, batch_size)
        resolved = []
        for item in values:
            if _is_path_like(item):
                resolved.append(self.load_path(item))
            else:
                resolved.append(item)
        return resolved

    def _build_task_label_views(self, shared_labels: List[Any]) -> Dict[str, List[Any]]:
        views: Dict[str, List[Any]] = {}
        for task, task_cfg in self.cfg['tasks'].items():
            label_keys = task_cfg.get('label_keys') or []
            if not label_keys:
                views[task] = [None for _ in shared_labels]
                continue
            task_values = []
            for label_obj in shared_labels:
                if not isinstance(label_obj, dict):
                    task_values.append(None)
                else:
                    task_values.append({key: label_obj.get(key) for key in label_keys})
            views[task] = task_values
        return views

    def __call__(self, *, batch_size: int, batch_kwargs: Dict[str, Any], raw_targets: Dict[str, Any]) -> Dict[str, Any]:
        shared_labels = self._resolve_shared_labels(batch_kwargs, batch_size)
        resolved_targets = {
            task: self._resolve_task_target(raw_targets.get(task), batch_size) for task in self.cfg['tasks']
        }
        return {
            'path_key': self.path_key,
            'shared_labels': shared_labels,
            'task_targets': resolved_targets,
            'task_label_views': self._build_task_label_views(shared_labels),
        }
