from torch import nn


class VGGTUpsampler(nn.Module):

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise TypeError(f'VGGTUpsampler expects a dict-like config, but got {type(cfg)!r}.')
        merged_cfg = dict(cfg)
        merged_cfg.update(kwargs)
        self.cfg = merged_cfg

    def forward(self, *input, **kwargs):
        raise NotImplementedError('Please provide the concrete VGGT upsampler implementation.')


VggtUpsampler = VGGTUpsampler
