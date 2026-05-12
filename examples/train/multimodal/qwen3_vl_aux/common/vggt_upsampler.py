from torch import nn


class VGGTUpsampler(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, *input, **kwargs):
        raise NotImplementedError('Please provide the concrete VGGT upsampler implementation.')


VggtUpsampler = VGGTUpsampler
