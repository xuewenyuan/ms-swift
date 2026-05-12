import copy
from typing import Any, Dict

from torch import nn

from aux_task.bev.bev_aux_loss import BevAuxLoss
from aux_task.cog.cog_aux_loss import CogAuxLoss
from aux_task.god.god_aux_loss import GodAuxLoss
from config_utils import to_config_node


AUX_LOSS_REGISTRY = {
    'bev': BevAuxLoss,
    'cog': CogAuxLoss,
    'god': GodAuxLoss,
}


def build_aux_loss(task_name: str, cfg: Dict[str, Any]) -> nn.Module:
    try:
        loss_cls = AUX_LOSS_REGISTRY[task_name]
    except KeyError as exc:
        raise KeyError(f'Unknown auxiliary loss task: {task_name!r}.') from exc
    task_cfg = copy.deepcopy(cfg['tasks'][task_name])
    loss_cfg = copy.deepcopy(task_cfg.get('loss', {}))
    loss_cfg.setdefault('task_name', task_name)
    loss_cfg.setdefault('shared_cfg', copy.deepcopy(cfg.get('shared', {})))
    loss_cfg.setdefault('task_cfg', task_cfg)
    loss_cfg.setdefault('plugin_cfg', cfg)
    return loss_cls(to_config_node(loss_cfg))


class AuxSeparateLosses(nn.Module):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.active_tasks = [
            task for task, task_cfg in cfg['tasks'].items() if task_cfg.get('enabled', True)
        ]
        self.losses = nn.ModuleDict({
            task: build_aux_loss(task, cfg)
            for task in self.active_tasks
        })

    def forward(self, task_name: str, *input, **kwargs):
        if task_name not in self.losses:
            raise KeyError(f'Auxiliary loss for task "{task_name}" is not enabled.')
        predictions = kwargs.get('predictions')
        targets = kwargs.get('targets')
        payload = {f'{task_name}_aux_head_output': predictions}
        return self.losses[task_name](targets, payload)


AuxTaskLosses = AuxSeparateLosses
