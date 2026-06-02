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


def _has_env(env_key: str) -> bool:
    return env_key in os.environ and os.environ[env_key] != ''


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    shared_cfg = config.setdefault('shared', {})
    if _has_env('QWEN3VL_AUX_HIDDEN_LAYERS'):
        shared_cfg['layer_indices'] = _parse_int_list(os.environ['QWEN3VL_AUX_HIDDEN_LAYERS'])
    if _has_env('QWEN3VL_AUX_MERGE_SIZE'):
        shared_cfg['merge_size'] = int(os.environ['QWEN3VL_AUX_MERGE_SIZE'])
    elif _has_env('SPATIAL_MERGE_SIZE'):
        shared_cfg['merge_size'] = int(os.environ['SPATIAL_MERGE_SIZE'])
    if _has_env('QWEN3VL_AUX_LM_WEIGHT'):
        shared_cfg['lm_loss_weight'] = float(os.environ['QWEN3VL_AUX_LM_WEIGHT'])
    if _has_env('QWEN3VL_AUX_ENABLED_TASKS'):
        shared_cfg['enabled_tasks'] = _parse_task_list(os.environ['QWEN3VL_AUX_ENABLED_TASKS'])

    label_loader_cfg = shared_cfg.setdefault('label_loader', {})
    if _has_env('QWEN3VL_AUX_LABEL_PATH_KEY'):
        label_loader_cfg['path_key'] = os.environ['QWEN3VL_AUX_LABEL_PATH_KEY']
    if _has_env('QWEN3VL_AUX_LABEL_FORMAT'):
        label_loader_cfg['format'] = os.environ['QWEN3VL_AUX_LABEL_FORMAT']
    if _has_env('QWEN3VL_AUX_LABEL_CACHE_SIZE'):
        label_loader_cfg['cache_size'] = int(os.environ['QWEN3VL_AUX_LABEL_CACHE_SIZE'])

    upsampler_cfg = shared_cfg.setdefault('upsampler_cfg', {})
    if _has_env('QWEN3VL_AUX_UPSAMPLER_TYPE'):
        upsampler_cfg['type'] = os.environ['QWEN3VL_AUX_UPSAMPLER_TYPE']
    if _has_env('QWEN3VL_AUX_DPT_PATCH_SIZE'):
        upsampler_cfg['patch_size'] = int(os.environ['QWEN3VL_AUX_DPT_PATCH_SIZE'])
    if _has_env('QWEN3VL_AUX_DPT_FEATURES'):
        upsampler_cfg['features'] = int(os.environ['QWEN3VL_AUX_DPT_FEATURES'])
    if _has_env('QWEN3VL_AUX_DPT_OUT_CHANNELS'):
        upsampler_cfg['out_channels'] = _parse_int_list(os.environ['QWEN3VL_AUX_DPT_OUT_CHANNELS'])

    tasks_cfg = config.setdefault('tasks', {})
    for task in AUX_TASKS:
        task_cfg = tasks_cfg.setdefault(task, {})
        if _has_env(f'QWEN3VL_{task.upper()}_LOSS_WEIGHT'):
            loss_weight = float(os.environ[f'QWEN3VL_{task.upper()}_LOSS_WEIGHT'])
            task_cfg['train_weight'] = loss_weight
            task_cfg['loss_weight'] = loss_weight
        if _has_env(f'QWEN3VL_{task.upper()}_NUM_CLASSES'):
            task_cfg.setdefault('head', {})['num_classes'] = int(os.environ[f'QWEN3VL_{task.upper()}_NUM_CLASSES'])
    return config


def get_aux_head_lr(default_lr: float) -> float:
    return float(os.environ.get('QWEN3VL_AUX_HEAD_LR', str(default_lr)))


def apply_aux_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = _apply_env_overrides(copy.deepcopy(config))
    enabled_tasks = merged.get('shared', {}).get('enabled_tasks')
    if enabled_tasks is not None:
        enabled_set = set(enabled_tasks)
        for task in AUX_TASKS:
            merged['tasks'][task]['enabled'] = task in enabled_set
    merged['layer_indices'] = list(merged['shared']['layer_indices'])
    merged['hidden_size'] = merged['shared']['hidden_size']
    return merged


def build_aux_config(target_model) -> Dict[str, Any]:
    text_config = getattr(target_model.config, 'text_config', target_model.config)
    hidden_size = getattr(text_config, 'hidden_size', None) or getattr(target_model.config, 'hidden_size')
    has_enabled_tasks_env = _has_env('QWEN3VL_AUX_ENABLED_TASKS')
    enabled_tasks_env = set(_parse_task_list(os.environ.get('QWEN3VL_AUX_ENABLED_TASKS', '')))
    default_bev_area = [[-45.4, 95.4], [-44.8, 44.8]]
    default_common_god_area = [[-19.8, 95.4], [-22.4, 22.4]]
    default_common_cog_area = [[-16.0, 95.4], [-22.4, 22.4]]
    default_god_resolution = 0.8
    default_foundation_resolution = 3.2
    default_patch_size = int(default_foundation_resolution // default_god_resolution)
    base_config: Dict[str, Any] = {
        'shared': {
            'hidden_size': hidden_size,
            'layer_indices': _parse_int_list(os.environ.get('QWEN3VL_AUX_HIDDEN_LAYERS', '6,13,20,27')),
            'merge_size': int(os.environ.get('QWEN3VL_AUX_MERGE_SIZE', os.environ.get('SPATIAL_MERGE_SIZE', '2'))),
            'lm_loss_weight': float(os.environ.get('QWEN3VL_AUX_LM_WEIGHT', '0.0')),
            'enabled_tasks': list(enabled_tasks_env) if has_enabled_tasks_env else None,
            'bev_area': default_bev_area,
            'god_resolution': default_god_resolution,
            'foundation_resolution': default_foundation_resolution,
            'common_cog_area': default_common_cog_area,
            'stage_dims': [128, 256, 512, hidden_size],
            'label_loader': {
                'path_key': os.environ.get('QWEN3VL_AUX_LABEL_PATH_KEY', 'labels_path'),
                'format': os.environ.get('QWEN3VL_AUX_LABEL_FORMAT', 'mspack'),
                'src_label_path': [2, 0],
                'cache_size': int(os.environ.get('QWEN3VL_AUX_LABEL_CACHE_SIZE', '256')),
            },
            'upsampler_cfg': {
                'type': os.environ.get('QWEN3VL_AUX_UPSAMPLER_TYPE', 'vggt_upsampler'),
                'patch_size': int(os.environ.get('QWEN3VL_AUX_DPT_PATCH_SIZE', str(default_patch_size))),
                'features': int(os.environ.get('QWEN3VL_AUX_DPT_FEATURES', '256')),
                'out_channels': _parse_int_list(
                    os.environ.get('QWEN3VL_AUX_DPT_OUT_CHANNELS', f'128,256,512,{hidden_size}')),
            },
        },
        'tasks': {},
    }
    for task in AUX_TASKS:
        base_config['tasks'][task] = {
            'enabled': task in enabled_tasks_env if has_enabled_tasks_env else True,
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

    base_config['tasks']['god']['head'].update({
        'in_channels': 128,
        'out_channels': 128,
        'conv_layer_num': 4,
        'god_bev_area': default_common_god_area,
        'common_bev_area': default_common_god_area,
        'god_resolution': 0.8,
        'god_distill': True,
        'head_key': ['occupancy', 'semantic', 'visibility', 'drivable'],
        'head_cls_num': {
            'occupancy': 16,
            'semantic': 112,
            'visibility': 16,
            'drivable': 1,
        },
        'head_sigmoid': {
            'occupancy': True,
            'semantic': False,
            'visibility': True,
            'drivable': True,
        },
    })
    base_config['tasks']['god']['loss'].update({
        'hist_seq_len': 0,
        'semantic_class': 7,
        'decouple_sem': True,
        'loss_tasks': ['occupancy', 'semantic', 'visibility', 'drivable'],
        'occupancy_weights': [4.0, 4.0, 4.0, 3.0, 2.5, 1.0, 4.0],
        'semantic_weights': [1.3, 3.0, 2.0, 1.0, 2.8, 0.05, 3.0],
        'visibility_weights': [2.0, 1.0],
        'drivable_weights': [1.0, 2.0],
        'loss_weight': 5.0,
        'god_distill': True,
    })

    base_config['tasks']['bev']['head'].update({
        'in_channels': 128,
        'out_channels': 64,
        'conv_layer_num': 2,
        'head_key': ['heatmap', 'reg', 'height', 'dim', 'rot', 'vel', 'movement'],
        'common_heads': ['reg', 'height', 'dim', 'rot', 'vel', 'movement'],
        'head_cls_num': {
            'heatmap': 4,
            'reg': 2,
            'height': 1,
            'dim': 3,
            'rot': 2,
            'vel': 2,
            'movement': 1,
        },
        'tasks': [{'class_names': ['vehicle', 'pedestrian', 'cyclist', 'other']}],
        'bev_distill': True,
        'bbox_coder': {
            'pc_range': [-45.4, -44.8, -5.0],
            'post_center_range': [-45.4, -44.8, -5.0, 95.4, 44.8, 5.0],
            'max_num': 500,
            'score_threshold': 0.25,
            'out_size_factor': 1,
            'voxel_size': [0.4, 0.4, 0.5],
            'code_size': 9,
        },
        'norm_bbox': True,
    })
    base_config['tasks']['bev']['loss'].update({
        'hist_seq_len': 0,
        'loss_weight': 1.0,
        'bbox_weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 0.3, 0.3],
        'bev_distill': True,
        'cls_cfg': {
            'reduction': 'mean',
        },
        'bbox_cfg': {
            'reduction': 'mean',
            'loss_weight': 0.25,
        },
        'bbox_coder': {
            'pc_range': [-45.4, -44.8, -5.0],
            'post_center_range': [-45.4, -44.8, -5.0, 95.4, 44.8, 5.0],
            'max_num': 500,
            'out_size_factor': 1,
            'voxel_size': [0.4, 0.4, 0.5],
            'code_size': 9,
        },
        'post_processing': {
            'activate': True,
            'dedup_head': True,
            '_lidar_setup': '3lidar',
            'score_thresh': 0.1,
            'nms_type': 'nms_2d_npu',
            'nms_thresh': 0.1,
            'nms_head_thrs': 0.4,
            'nms_pre_maxsize': 4096,
            'nms_post_maxsize': 100,
            'nms_on_sight': [0.0, 0.0, 0.45, 0.45, 0.0, 0.0],
            'thrs_vision_area': [0.25, 0.20, 0.27, 0.23, 0.25, 0.2],
            'thrs_lidar_area': [0.25, 0.20, 0.27, 0.23, 0.25, 0.2],
        },
    })

    user_config = _load_user_config()
    merged = _deep_update(copy.deepcopy(base_config), user_config)
    return apply_aux_env_overrides(merged)
