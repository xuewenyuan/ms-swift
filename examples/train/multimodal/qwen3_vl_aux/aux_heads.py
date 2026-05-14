from contextlib import contextmanager
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

from aux_task.bev.bev_aux_head import BevAuxHead
from aux_task.cog.cog_aux_head import CogAuxHead
from aux_task.god import GodAuxHead
from common.token_feature_adapter import Qwen3VLAuxTokenFeatureAdapter, crop_bev_area_a2b
from common.unet_decoder import UnetDecoderDeconv, UnetDecoderInterpolate, UpDeconvSimple
from common.vggt_upsampler import VggtUpsampler
from config_utils import to_config_node

try:  # pragma: no cover - optional dependency
    from rdcog.utils.timer import synctimer as timer
except ImportError:  # pragma: no cover - fallback for local/dev environments

    class _NoOpTimer:

        @contextmanager
        def ticktock(self, *_args, **_kwargs):
            yield

    timer = _NoOpTimer()


TASK_ORDER = ('god', 'cog', 'bev')
AUX_HEAD_REGISTRY = {
    'bev': BevAuxHead,
    'cog': CogAuxHead,
    'god': GodAuxHead,
}

def _get_nested(cfg: Mapping[str, Any], *keys: str, default=None):
    current = cfg
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _build_task_head(task_name: str, cfg: Dict[str, Any]) -> nn.Module:
    head_cls = AUX_HEAD_REGISTRY[task_name]
    task_cfg = dict(_get_nested(cfg, 'tasks', task_name, 'head', default={}) or {})
    task_cfg.setdefault('task_name', task_name)
    return head_cls(to_config_node(task_cfg))


class AuxSeparateHeads(nn.Module):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        tasks_cfg = cfg.get('tasks', {})
        self.active_tasks = [task for task in TASK_ORDER if tasks_cfg.get(task, {}).get('enabled', True)]

        self.bev_area = _get_nested(cfg, 'shared', 'bev_area', default=cfg.get('god_bev_area', [[-19.8, 95.4], [-22.4, 22.4]]))
        self.resolution = _get_nested(cfg, 'shared', 'god_resolution', default=cfg.get('god_resolution', 0.4))
        self.foundation_resolution = _get_nested(
            cfg, 'shared', 'foundation_resolution', default=cfg.get('foundation_resolution', 6.4))
        self.intermediate_layer_idx = list(
            _get_nested(cfg, 'shared', 'layer_indices', default=cfg.get('intermediate_layer_idx', [6, 13, 20, 27])))
        self.scale = int(self.foundation_resolution // self.resolution)

        upsampler_cfg = _get_nested(cfg, 'shared', 'upsampler_cfg', default={}) or {}
        self.stage_dims = list(
            _get_nested(cfg, 'shared', 'stage_dims',
                        default=upsampler_cfg.get('out_channels', cfg.get('stage_dims', [128, 256, 512, 1024]))))
        self.up_sample_type = upsampler_cfg.get('type', cfg.get('up_sample_type', 'updeconv'))

        # Kept for parity with the original implementation. Re-enable if the
        # exported inference or cache-dump path is wired back in later.
        # self.auxiliary_task_num = len(self.active_tasks)
        # self.onnx = cfg.get('onnx', False)
        # self.dump_cache = cfg.get('dump_cache', False)

        self.unets = nn.ModuleDict({
            task: self._build_unet(upsampler_cfg) for task in self.active_tasks
        })
        self.task_heads = nn.ModuleDict({
            task: _build_task_head(task, cfg) for task in self.active_tasks
        })

        common_cog_area = _get_nested(
            cfg, 'shared', 'common_cog_area', default=cfg.get('common_cog_area', [[-16, 95.4], [-22.4, 22.4]]))
        self.seg_crop_lft, self.seg_crop_rht, self.seg_crop_top, self.seg_crop_down = crop_bev_area_a2b(
            self.bev_area, common_cog_area, self.resolution)

        self._auto_init()
        if 'bev' in self.task_heads and hasattr(self.task_heads['bev'], 'init_heatmap_weights'):
            self.task_heads['bev'].init_heatmap_weights()
        self.token_feature_adapter = Qwen3VLAuxTokenFeatureAdapter(
            intermediate_layer_idx=self.intermediate_layer_idx,
            merge_size=_get_nested(cfg, 'shared', 'merge_size', default=2),
        )
        # Historical fixed token-length slicing is intentionally no longer used.
        # self.pv_token_len = cfg.get('pv_token_len', 740)
        # self.bev_token_len = cfg.get('bev_token_len', 504)
        # self.navi_token_len = cfg.get('navi_token_len', 32)

    def _auto_init(self):
        pass

    def _build_unet(self, upsampler_cfg: Mapping[str, Any]) -> nn.Module:
        if self.up_sample_type == 'interpolate':
            return UnetDecoderInterpolate(self.stage_dims)
        if self.up_sample_type == 'deconv':
            return UnetDecoderDeconv(self.stage_dims)
        if self.up_sample_type == 'updeconv':
            return UpDeconvSimple(self.stage_dims)
        if self.up_sample_type == 'vggt_upsampler':
            return VggtUpsampler(
                dim_in=upsampler_cfg.get('dim_in', self.stage_dims[-1]),
                patch_size=upsampler_cfg.get('patch_size', self.scale),
                features=upsampler_cfg.get('features', self.stage_dims[0]),
                out_channels=upsampler_cfg.get('out_channels', self.stage_dims),
                intermediate_layer_idx=upsampler_cfg.get('intermediate_layer_idx', self.intermediate_layer_idx),
                pos_embed=upsampler_cfg.get('pos_embed', False),
                feature_only=upsampler_cfg.get('feature_only', True),
            )
        raise NotImplementedError(f'up_sample_type {self.up_sample_type} not implemented')

    def forward(self, *inputs, **kwargs):
        payload = self.token_feature_adapter.normalize_head_inputs(inputs)
        hidden_state_payload, fusion_feat, height, width = self.token_feature_adapter.build_fusion_features(
            payload['hidden_states'],
            payload['input_ids'],
            payload['image_grid_thw'],
            payload['model_config'],
        )
        batch_size = fusion_feat.shape[0]

        with timer.ticktock('-   unet'):
            unet_feats = {}
            if self.up_sample_type in {'interpolate', 'deconv', 'updeconv'}:
                for task, unet in self.unets.items():
                    unet_output = unet([fusion_feat])
                    if isinstance(unet_output, (list, tuple)):
                        unet_feats[task] = unet_output[-1]
                    else:
                        unet_feats[task] = unet_output
            elif self.up_sample_type == 'vggt_upsampler':
                images_template = torch.zeros(
                    (batch_size, 1, 3, int(height * self.scale), int(width * self.scale)),
                    device=fusion_feat.device,
                    dtype=fusion_feat.dtype,
                )
                for task, unet in self.unets.items():
                    unet_output = unet(
                        hidden_state_payload['image_tokens_list'],
                        images=images_template,
                        patch_start_idx=0,
                    )
                    if isinstance(unet_output, torch.Tensor) and unet_output.ndim == 5:
                        unet_feats[task] = unet_output[:, 0, :, :, :]
                    else:
                        unet_feats[task] = unet_output
            else:
                raise NotImplementedError(f'up_sample_type {self.up_sample_type} not implemented')

        output = {
            'unet_feats': unet_feats,
            'fusion': fusion_feat,
        }
        vis_dict = {
            'unet': list(unet_feats.values())[-1] if unet_feats else fusion_feat,
            'fusion': fusion_feat,
        }

        for task in self.active_tasks:
            head = self.task_heads[task]
            with timer.ticktock(f'-   {task} aux head'):
                task_output, task_feat = head(unet_feats[task])
            if task == 'cog':
                task_output = {
                    key: value if 'cog_neck' in key else value[:, :, self.seg_crop_top:self.seg_crop_down,
                                                                self.seg_crop_lft:self.seg_crop_rht]
                    for key, value in task_output.items()
                }
            output[f'{task}_aux_head_output'] = task_output
            if not self.training and task_feat is not None:
                vis_dict.update(task_feat)

        if not self.training:
            output['vis_dict'] = vis_dict
        return [output]


AuxTaskHeads = AuxSeparateHeads
