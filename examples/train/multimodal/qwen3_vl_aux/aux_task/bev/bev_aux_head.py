from aux_base import BaseModule


class BevAuxHead(BaseModule):

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, *input, **kwargs):
        raise NotImplementedError('Please provide the concrete Bev auxiliary head implementation.')
