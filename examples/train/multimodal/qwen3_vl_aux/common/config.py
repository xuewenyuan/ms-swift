import copy
import json
import os
from typing import Any, Dict, List, Mapping


AUX_TASKS = ('bev', 'cog', 'god')


def _parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(',') if item.strip()]


def _parse_task_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(',') if item.strip()]


def _load_user_config() -> Dict[str, Any]:
    config_path = os.environ.get('QWEN3VL_AUX_CONFIG_PATH')
    config_raw = os.environ.get('QWEN3VL_AUX_CONFIG')
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    if config_raw:
        return json.loads(config_raw)
    return {}


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def get_aux_head_lr(default_lr: float) -> float:
    return float(os.environ.get('QWEN3VL_AUX_HEAD_LR', str(default_lr)))


def build_aux_config(target_model) -> Dict[str, Any]:
    text_config = getattr(target_model.config, 'text_config', target_model.config)
    hidden_size = getattr(text_config, 'hidden_size', None) or getattr(target_model.config, 'hidden_size')
    enabled_tasks_env = set(_parse_task_list(os.environ.get('QWEN3VL_AUX_ENABLED_TASKS', '')))
    base_config: Dict[str, Any] = {
        'shared': {
            'hidden_size': hidden_size,
            'layer_indices': _parse_int_list(os.environ.get('QWEN3VL_AUX_HIDDEN_LAYERS', '6,13,20,27')),
            'merge_size': int(os.environ.get('QWEN3VL_AUX_MERGE_SIZE', os.environ.get('SPATIAL_MERGE_SIZE', '2'))),
            'lm_loss_weight': float(os.environ.get('QWEN3VL_AUX_LM_WEIGHT', '1.0')),
            'enabled_tasks': list(enabled_tasks_env) if enabled_tasks_env else list(AUX_TASKS),
            'label_loader': {
                'path_key': os.environ.get('QWEN3VL_AUX_LABEL_PATH_KEY', 'labels_path'),
                'format': os.environ.get('QWEN3VL_AUX_LABEL_FORMAT', 'mspack'),
                'src_label_path': [2, 0],
                'cache_size': int(os.environ.get('QWEN3VL_AUX_LABEL_CACHE_SIZE', '256')),
            },
            'upsampler_cfg': {
                'patch_size': int(os.environ.get('QWEN3VL_AUX_DPT_PATCH_SIZE', '32')),
                'features': int(os.environ.get('QWEN3VL_AUX_DPT_FEATURES', '256')),
                'out_channels': _parse_int_list(os.environ.get('QWEN3VL_AUX_DPT_OUT_CHANNELS', '256,512,1024,1024')),
            },
        },
        'tasks': {},
    }
    for task in AUX_TASKS:
        base_config['tasks'][task] = {
            'enabled': task in enabled_tasks_env if enabled_tasks_env else True,
            'label_key': f'y_{task}',
            'label_keys': [],
            'loss_weight': float(os.environ.get(f'QWEN3VL_{task.upper()}_LOSS_WEIGHT', '1.0')),
            'head': {
                'task_name': task,
                'num_classes': int(os.environ.get(f'QWEN3VL_{task.upper()}_NUM_CLASSES', '2')),
            },
            'loss': {
                'task_name': task,
            },
        }

    user_config = _load_user_config()
    merged = _deep_update(copy.deepcopy(base_config), user_config)
    enabled_tasks = merged.get('shared', {}).get('enabled_tasks')
    if enabled_tasks:
        enabled_set = set(enabled_tasks)
        for task in AUX_TASKS:
            merged['tasks'][task]['enabled'] = task in enabled_set
    merged['layer_indices'] = list(merged['shared']['layer_indices'])
    merged['hidden_size'] = merged['shared']['hidden_size']
    return merged
