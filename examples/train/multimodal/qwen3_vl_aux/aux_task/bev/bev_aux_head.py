import torch
import torch.nn as nn

from infrastructure import BaseModule
from common.common_module import ConvClassifier, LiteClsHead
# from lib.visualizer.train_watcher.feature_train_watcher import FeatureWatcher
from .centerpoint_bbox_coders import CenterPointBBoxCoder
from .utils import Obj

class BevAuxHead(BaseModule):
    def __init__(self, cfg):
        super(BevAuxHead, self).__init__(cfg)
        in_channels = cfg.get('in_channels', 128)
        out_channels = cfg.get('out_channels', 32)
        conv_layer_num = cfg.get('conv_layer_num', 4)

        self.init_bias = cfg.get('init_bias', -4.5951)
        
        self.upsample = nn.Sequential(
            # Scale factor 2 doubles the size
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # Padding 1 preserves dimensions after convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.bev_distill = cfg.get('bev_distill', False)
        if self.bev_distill:
            self.distill_head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True),
            )

        self.feat_conv = ConvClassifier(in_channels, in_channels, layer_num=conv_layer_num)

        self.head_key = cfg.get('head_key', ['reg', 'height', 'dim'])
        self.head_cls_num = cfg.get('head_cls_num')

        self.prediction_head = nn.ModuleDict()
        for key in self.head_key:
            assert key in self.head_cls_num, f"{key} not in head_cls_num"

            cls_num = self.head_cls_num[key]
            self.prediction_head[key] = LiteClsHead(in_channels, out_channels, cls_num)

        if self.training == False:
            bbox_coder_cfg = cfg["bbox_coder"]
            self.bbox_coder = CenterPointBBoxCoder(**bbox_coder_cfg)
            self.code_dim = 12 if 'vel' in cfg.common_heads else 10
            self.code_dim += 1 if 'movement' in cfg.common_heads else 0
            self.code_dim += 50 if 'traj' in cfg.common_heads else 0
            self.norm_bbox = cfg.get('norm_bbox', True)
            num_classes = [len(t['class_names']) for t in cfg["tasks"]]
            self.num_classes = num_classes

    def forward(self, *inputs, **kwargs):
        output = dict()
        
        up_feat = self.upsample(inputs[0]) # [2, 128, 288, 112]

        dense_feat = self.feat_conv(up_feat) # [2, 128, 288, 112]
        vis_feat = {
            'bev_neck': dense_feat,
        }
        output['bev_det_feat'] = dense_feat

        if self.bev_distill:
            distill_feat = self.distill_head(dense_feat)
            output['distill_feat'] = distill_feat
            vis_feat['bev_distill'] = distill_feat

        for key in self.head_key:
            pred = self.prediction_head[key](dense_feat)
            output[key] = pred

        if self.training == False:
            return output, vis_feat
        
        # try: 
        #     FeatureWatcher.watch(vis_feat)
        # except Exception as e:
        #     logger.warning(f"{e}")

        return output, None

    def init_heatmap_weights(self):
        """Initialize weights."""
        for head in self.head_key:
            if head in ('heatmap'):
                heatmap_conv = self.prediction_head['heatmap'].classifier[-1]
                heatmap_conv.bias.data.fill_(self.init_bias)
                nn.init.normal_(heatmap_conv.weight, mean=0, std=0.01)

    def get_bboxes(self, preds_dicts, return_origin_bbox_preds=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        batch_size = preds_dicts[0]['heatmap'].shape[0]
        preds = preds_dicts[0]['heatmap'].new_ones(
            [batch_size, self.bbox_coder.max_num * len(preds_dicts), self.code_dim]) * -1

        for task_id, preds_dict in enumerate(preds_dicts):
            batch_heatmap = preds_dict['heatmap'].sigmoid()

            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict['dim'])
            else:
                batch_dim = preds_dict['dim']

            batch_rots = preds_dict['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']
            else:
                batch_vel = None

            ext_attrs = []

            if 'movement' in preds_dict:
                ext_attrs.insert(0, preds_dict['movement'].sigmoid())

            if len(ext_attrs) > 0:
                ext_attrs = torch.cat(ext_attrs, dim=1)

            temp, origin_bbox_preds = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id, ext_attrs=ext_attrs)

            # concat each batch
            for batch, pred in enumerate(temp):
                flag = sum(self.num_classes[:task_id])  # add class offset
                start = self.bbox_coder.max_num * task_id
                end = start + pred['bboxes'].shape[0]
                preds[batch, start:end, :Obj.ry.value + 1] = pred['bboxes'][:, :Obj.ry.value + 1]
                preds[batch, start:end, Obj.label.value] = pred['labels'] + flag
                preds[batch, start:end, Obj.conf.value] = pred['scores']

                assert batch_vel is not None
                attr_num = pred['bboxes'].shape[-1] - 7  # 10 - 7 = 3
                assert (preds.shape[-1] - 10) == attr_num  # 13 - 10 = 3
                preds[batch, start:end, -attr_num:] = pred['bboxes'][:, -attr_num:]

        if return_origin_bbox_preds:
            return preds, origin_bbox_preds

        return preds
