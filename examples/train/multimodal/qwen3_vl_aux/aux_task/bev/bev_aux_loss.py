import torch
from torch import nn
from loguru import logger
from .utils import INVALID_VELOCITY
from ..losses import GaussianFocalLoss, L1Loss
from infrastructure import loss_formatter
# from lib.visualizer.train_watcher.bev_watcher import BEVWatcher
# from lib.visualizer.train_watcher.distillation_watcher import DistillationWatcher
from .centerpoint_bbox_coders import CenterPointBBoxCoder
# from lib.utils.error_monitor.monitor import train_abnormal_counter as counter
from uvp_module.models.utils.det_utils import PostProrcessor

def clip_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.clamp(x.sigmoid(), min=eps, max=1 - eps)


class BevAuxLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.hist_seq_len = cfg.get("hist_seq_len", 0) + 1

        self.bbox_weight = cfg.get('bbox_weight', None)
        self.cls_loss_weight = cfg.get('cls_loss_weight', None)
        self.loss_weight = cfg.get('loss_weight', None)
        self.bev_distill = cfg.get('bev_distill', False)
        cls_cfg = cfg.get('cls_cfg', None)
        self.loss_cls = GaussianFocalLoss(**cls_cfg)

        bbox_cfg = cfg.get('bbox_cfg', None)
        self.loss_bbox = L1Loss(**bbox_cfg)

        self.loss_movement = nn.BCEWithLogitsLoss(reduction='none')

        bbox_coder = cfg.get('bbox_coder', None)
        self.bbox_decoder = CenterPointBBoxCoder(**bbox_coder)

        self.post_process = PostProrcessor(cfg) if cfg.post_processing.activate else None

        if self.bev_distill:
            self.loss_distill = nn.MSELoss(reduction='none')

    def _normalize_gt_layout(self, gt):
        if gt is None:
            raise ValueError('centerpoint_head_obj_white is missing from auxiliary labels.')
        if not isinstance(gt, (list, tuple)):
            raise TypeError(
                f'centerpoint_head_obj_white must be a list/tuple, got {type(gt)!r}.')
        if not gt:
            raise ValueError('centerpoint_head_obj_white is empty.')

        # Some upstream collation paths hand us a field-major layout:
        #   [[heatmaps_b0, ...], [anno_boxes_b0, ...], ...]
        # while the original loss expects a sample-major layout:
        #   [[heatmaps, anno_boxes, ...], [heatmaps, anno_boxes, ...], ...]
        # Detect the common field-major form (outer length 8) and transpose it back.
        if len(gt) == 8 and all(isinstance(item, (list, tuple)) for item in gt):
            gt = [list(sample) for sample in zip(*gt)]

        normalized = [list(sample) if isinstance(sample, tuple) else sample for sample in gt]
        if not normalized:
            raise ValueError('centerpoint_head_obj_white is empty after normalization.')
        first = normalized[0]
        if not isinstance(first, (list, tuple)) or len(first) < 8:
            raise ValueError(
                'centerpoint_head_obj_white has an unexpected layout after normalization: '
                f'expected sample entries with at least 8 fields, got {type(first)!r} '
                f'with length {len(first) if isinstance(first, (list, tuple)) else "n/a"}')
        return normalized

    def get_target(self, gt):
        gt = self._normalize_gt_layout(gt)

        def make_concats(yyy):
            yyy = list(map(list, zip(*yyy)))
            return [torch.stack(xxx) for xxx in yyy]
        
        heatmaps, anno_boxes, inds, masks, _, _, movement, movement_weight = list(map(list, zip(*gt)))

        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]

        heatmaps = make_concats(heatmaps)
        movement = make_concats(movement)
        movement_weight = make_concats(movement_weight)

        return heatmaps[0], anno_boxes[0], inds[0], masks[0], movement[0], movement_weight[0]

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat
    
    def forward(self, *inputs, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        inputs = list(inputs)
        uvp_labels = inputs.pop(0)
        labels =  [[]]
        labels[0] = uvp_labels
        pred_dicts = inputs[0]['bev_aux_head_output']
        # pred_dicts = {
        #     'bev_det_feat': torch.zeros((2, 128, 288, 112), device='cuda'),
        #     'distill_feat': torch.zeros((2, 64, 144, 56), device='cuda'),
        #     'heatmap': torch.zeros((2, 4, 288, 112), device='cuda'),
        #     'reg': torch.zeros((2, 2, 288, 112), device='cuda'),
        #     'height': torch.zeros((2, 1, 288, 112), device='cuda'),
        #     'dim': torch.zeros((2, 3, 288, 112), device='cuda'),
        #     'rot': torch.zeros((2, 2, 288, 112), device='cuda'),
        #     'vel': torch.zeros((2, 2, 288, 112), device='cuda'),
        #     'movement': torch.zeros((2, 1, 288, 112), device='cuda'),
            
        # }
        # gt_dict = labels[0].get('centerpoint_head_gt_rl', None)
        gt_dict = labels[0].get('centerpoint_head_obj_white', None)
        gt_heatmaps, gt_boxes, inds, masks, gt_movement, movement_weight = self.get_target(gt_dict)

        pred_heatmap = clip_sigmoid(pred_dicts['heatmap'])

        sample_mask = labels[0].get('rl_sample_mask', None)
        if sample_mask is not None:
            gt_heatmaps = gt_heatmaps * sample_mask[..., None, None]
        num_pos = gt_heatmaps.eq(1).float().sum().item()

        bs, c, h, w = pred_heatmap.shape
        nearby_mask = torch.ones((1, 1, h, w), dtype=torch.float32).to(pred_heatmap.device)

        bevseg_valid = torch.ones((bs, 1), device=pred_heatmap.device).bool()
        if 'bevseg_cache_valid' in labels[0]:
            bevseg_valid = bevseg_valid & labels[0]['bevseg_cache_valid']

        # counter.count_error('bevseg_invalid_frames', (~bevseg_valid).sum().item())

        bev_feat_valid = torch.ones((bs, 1), device=sample_mask.device).bool()
        if 'bev_feat_cache_valid' in labels[0]:
            bev_feat_valid = bev_feat_valid & labels[0]['bev_feat_cache_valid']
        # counter.count_error('bev_feat_invalid_frames', (~bev_feat_valid).sum().item())

        not_cls_only = 1 - labels[0]['is_cls_only']
        cls_mask = sample_mask * bevseg_valid
        distill_mask = cls_mask * bev_feat_valid

        loss_heatmap = self.loss_cls(
            pred_heatmap,
            gt_heatmaps,
            nearby_mask=nearby_mask,
            weight=self.cls_loss_weight,
            avg_factor=max(num_pos, 1),
            sample_mask=cls_mask)

        # Regression loss for dimension, offset, height, rotation
        if sample_mask is None:
            num = (masks * not_cls_only).float().sum()
        else:
            num = (masks * not_cls_only * sample_mask).float().sum()

        mask = masks.unsqueeze(2).expand_as(gt_boxes).float()
        isnotnan = (~torch.isnan(gt_boxes)).float()
        mask *= isnotnan

        valid_velocity = (gt_boxes[..., -2:] < INVALID_VELOCITY).float()
        mask[..., -2:] *= valid_velocity

        bbox_weights = mask * mask.new_tensor(self.bbox_weight)

        pred_boxes = torch.cat((pred_dicts['reg'], 
                                pred_dicts['height'],
                                pred_dicts['dim'], 
                                pred_dicts['rot'],
                                pred_dicts['vel']), dim=1)
        pred_boxes = pred_boxes.permute(0, 2, 3, 1).flatten(1, 2)
        pred_boxes = self._gather_feat(pred_boxes, inds)
        
        if sample_mask is not None:
            reg_mask = not_cls_only * sample_mask

        reg_mask = reg_mask * bevseg_valid
        loss_bbox = self.loss_bbox(
            pred_boxes, 
            gt_boxes, 
            bbox_weights, 
            avg_factor=(num + 1e-4),
            sample_mask=reg_mask, 
            keep_last_dim=True)
        loss_bbox = loss_bbox.sum()

        pred_mov = pred_dicts['movement'].permute(0, 2, 3, 1).flatten(1, 2)
        pred_mov = self._gather_feat(pred_mov, inds)

        mov_weight = mask[..., -1, None] * movement_weight * not_cls_only[..., None] * bevseg_valid[..., None]
        if sample_mask is not None:
            mov_weight = mov_weight * sample_mask[..., None]

        valid_num = max(1.0, (mov_weight > 0).float().sum().item())
        loss_movement = self.loss_movement(pred_mov, gt_movement)
        loss_movement = (loss_movement * mov_weight).sum() / valid_num

        if self.bev_distill:
            gt_feat = labels[0]['pnc_bev_input']['dense_bev_feat']
            pred_feat = pred_dicts['distill_feat']
            distill_loss = self.loss_distill(pred_feat, gt_feat)
            distill_loss = distill_loss * distill_mask[..., None, None]
 
            # try:
            #     DistillationWatcher.watch(pred_feat, gt_feat, name='bev_distill')
            # except Exception as e:
            #     logger.warning(f"{e}")

        loss_dict = {
            'loss_heatmap': loss_heatmap, 
            'loss_bbox': loss_bbox, 
            'loss_movement': loss_movement, 
            'distill_loss': torch.mean(distill_loss)
            }

        origin_loss = []
        loss_weight = []
        weighted_loss = []

        for k, v in loss_dict.items():
            weight = self.loss_weight
            if k == 'distill_loss':
                weight = self.loss_weight / 3
 
            origin_loss.append(v)
            loss_weight.append(weight)
            weighted_loss.append(v*weight)

        final_loss = loss_formatter(
            weighted_loss=weighted_loss,
            origin_loss=origin_loss,
            loss_weight=loss_weight,
        )

        # try:
        #     BEVWatcher.watch(labels[0]['obj_label_white'], pred_dicts, self.bbox_decoder, self.post_process)
        # except Exception as e:
        #     logger.warning(f"{e}")

        return final_loss
