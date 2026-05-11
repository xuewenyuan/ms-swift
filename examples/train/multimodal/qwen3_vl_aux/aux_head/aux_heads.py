from typing import Any, Dict

from torch import nn

from .bev_aux_head import BevAuxHead
from .cog_aux_head import CogAuxHead
from .god_aux_head import GodAuxHead


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
    return head_cls(cfg)


class AuxSeparateHeads(nn.Module):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.heads = nn.ModuleDict({
            task: build_aux_head(
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
        return self.heads[task_name](*input, **kwargs)


AuxTaskHeads = AuxSeparateHeads
