from aux_base import BaseModule


class CogAuxLoss(BaseModule):

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, *input, **kwargs):
        raise NotImplementedError('Please provide the concrete Cog auxiliary loss implementation.')
