import copy
from typing import Any, Dict

from torch import nn

from aux_task.bev.bev_aux_head import BevAuxHead
from aux_task.cog.cog_aux_head import CogAuxHead
from aux_task.god.god_aux_head import GodAuxHead
from common.vggt_upsampler import VGGTUpsampler
from config_utils import to_config_node


AUX_HEAD_REGISTRY = {
    'bev': BevAuxHead,
    'cog': CogAuxHead,
    'god': GodAuxHead,
}


def build_aux_head(task_name: str, cfg: Dict[str, Any]) -> nn.Module:
    try:
        head_cls = AUX_HEAD_REGISTRY[task_name]
    except KeyError as exc:
        raise KeyError(f'Unknown auxiliary head task: {task_name!r}.') from exc
    task_cfg = copy.deepcopy(cfg['tasks'][task_name])
    head_cfg = copy.deepcopy(task_cfg.get('head', {}))
    head_cfg.setdefault('task_name', task_name)
    head_cfg.setdefault('shared_cfg', copy.deepcopy(cfg.get('shared', {})))
    head_cfg.setdefault('task_cfg', task_cfg)
    head_cfg.setdefault('plugin_cfg', cfg)
    return head_cls(to_config_node(head_cfg))


def build_aux_upsampler(task_name: str, cfg: Dict[str, Any]) -> nn.Module:
    upsampler_cfg = copy.deepcopy(cfg.get('shared', {}).get('upsampler_cfg', {}))
    upsampler_cfg.setdefault('task_name', task_name)
    upsampler_cfg.setdefault('hidden_size', cfg.get('shared', {}).get('hidden_size'))
    upsampler_cfg.setdefault('layer_indices', list(cfg.get('shared', {}).get('layer_indices', [])))
    upsampler_cfg.setdefault('merge_size', cfg.get('shared', {}).get('merge_size'))
    upsampler_cfg.setdefault('shared_cfg', copy.deepcopy(cfg.get('shared', {})))
    upsampler_cfg.setdefault('task_cfg', copy.deepcopy(cfg['tasks'][task_name]))
    upsampler_cfg.setdefault('plugin_cfg', cfg)
    return VGGTUpsampler(to_config_node(upsampler_cfg))


class AuxSeparateHeads(nn.Module):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.active_tasks = [
            task for task, task_cfg in cfg['tasks'].items() if task_cfg.get('enabled', True)
        ]
        self.heads = nn.ModuleDict({
            task: build_aux_head(task, cfg)
            for task in self.active_tasks
        })
        self.upsamplers = nn.ModuleDict({
            task: build_aux_upsampler(task, cfg) for task in self.active_tasks
        })

    def forward(self, task_name: str, *input, **kwargs):
        if task_name not in self.heads:
            raise KeyError(f'Auxiliary head for task "{task_name}" is not enabled.')
        upsampled = self.upsamplers[task_name](*input, **kwargs)
        if isinstance(upsampled, tuple):
            return self.heads[task_name](*upsampled, **kwargs)
        if isinstance(upsampled, list):
            return self.heads[task_name](*upsampled, **kwargs)
        return self.heads[task_name](upsampled, **kwargs)


AuxTaskHeads = AuxSeparateHeads
