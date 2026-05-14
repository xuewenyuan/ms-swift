import torch.nn as nn
from infrastructure import BaseModule
from common.common_module import ConvClassifier, LiteClsHead, crop_bev_area_a2b
# from lib.visualizer.train_watcher.feature_train_watcher import FeatureWatcher


class GodAuxHead(BaseModule):
    def __init__(self, cfg):
        super(GodAuxHead, self).__init__(cfg)
        in_channels = cfg.get('in_channels', 128)
        out_channels = cfg.get('out_channels', 32)
        conv_layer_num = cfg.get('conv_layer_num', 4)
        self.god_distill = cfg.get('god_distill', False)

        self.feat_conv = ConvClassifier(in_channels, in_channels, layer_num=conv_layer_num)
        
        if self.god_distill:
            self.distill_head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True),
            )

        self.head_key = cfg.get('head_key', ['semantic', 'occupancy'])
        self.head_cls_num = cfg.get('head_cls_num')
        self.head_sigmoid = cfg.get('head_sigmoid')

        self.prediction_head = nn.ModuleDict()
        for key in self.head_key:
            assert key in self.head_cls_num, f"{key} not in head_cls_num"
            assert key in self.head_sigmoid, f"{key} not in head_sigmoid"

            cls_num = self.head_cls_num[key]
            is_sigmoid = self.head_sigmoid[key]
            self.prediction_head[key] = LiteClsHead(in_channels, out_channels, cls_num, sigmoid_out=is_sigmoid)

        god_bev_area = cfg.get('god_bev_area', [[-45.4, 95.4], [-44.8, 44.8]])       # [[-19.8, 95.4], [-22.4, 22.4]] &GOD_BEV_AREA
        common_bev_area = cfg.get('common_bev_area', [[-19.8, 95.4], [-22.4, 22.4]]) # [[-19.8, 95.4], [-22.4, 22.4]] &COMMON_GOD_HEAD_AREA
        god_resolution = cfg.get('god_resolution', 0.8)
        self.seg_crop_lft, self.seg_crop_rht, self.seg_crop_top, self.seg_crop_down = crop_bev_area_a2b(god_bev_area, common_bev_area, god_resolution)

    def forward(self, *inputs, **kwargs):
        """
        Args:
            inputs:
            **kwargs:

        Returns:
            god_feat (list[torch.Tensor])
        """

        output = dict()
        dense_feat = self.feat_conv(inputs[0])
        vis_feat = {
            'god_neck': dense_feat,
        }

        if self.god_distill:
            distill_feat = self.distill_head(dense_feat)[:, :, self.seg_crop_top:self.seg_crop_down, self.seg_crop_lft:self.seg_crop_rht] # [2, 128, 144, 56] -> [2, 64, 144, 56]
            output['distill_feat'] = distill_feat
            vis_feat['god_distill'] = distill_feat

        for key in self.head_key:
            pred = self.prediction_head[key](dense_feat, pred_factor=2)[:, :, self.seg_crop_top*2:self.seg_crop_down*2, self.seg_crop_lft*2:self.seg_crop_rht*2] # [2, 128, 144, 56] -> [2, C, 288, 112]
            output[key] = pred

        if self.training == False:
            return output, vis_feat
        
        # try: 
        #     FeatureWatcher.watch(vis_feat)
        # except Exception as e:
        #     logger.warning(f"{e}")

        return output, None 
