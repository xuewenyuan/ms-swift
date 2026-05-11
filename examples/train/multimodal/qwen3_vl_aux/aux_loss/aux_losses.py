from typing import Any, Dict

from torch import nn

from .bev_aux_loss import BevAuxLoss
from .cog_aux_loss import CogAuxLoss
from .god_aux_loss import GodAuxLoss


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
    return loss_cls(cfg)


class AuxSeparateLosses(nn.Module):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.losses = nn.ModuleDict({
            task: build_aux_loss(
                task,
                {
                    'task_name': task,
                    'shared': cfg.get('shared', {}),
                    'task': cfg['tasks'][task],
                    'plugin': cfg,
                })
            for task in cfg['tasks']
        })

    def forward(self, task_name: str, *input, **kwargs):
        return self.losses[task_name](*input, **kwargs)


AuxTaskLosses = AuxSeparateLosses
