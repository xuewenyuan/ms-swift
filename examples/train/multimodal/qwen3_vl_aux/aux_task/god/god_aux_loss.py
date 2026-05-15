# !/usr/bin/python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import copy
from loguru import logger

from .god_utils import GOD_SEMANTIC_TYPE_MAPPING
from ..losses.loss_utils import MultiFocalLoss, MaskedBCEWithLogitsLoss
from infrastructure import loss_formatter
# from lib.visualizer.train_watcher.god_head_watcher import GodHeadWatcher
# from lib.utils.error_monitor.monitor import train_abnormal_counter as counter


def _attach_loss_items(final_loss, loss_items, weighted_loss_items):
    payload = {
        'loss_items': loss_items,
        'weighted_loss_items': weighted_loss_items,
    }
    for key, value in payload.items():
        try:
            setattr(final_loss, key, value)
        except Exception:
            pass
        try:
            final_loss[key] = value
        except Exception:
            pass
    return final_loss


class GodAuxLoss(nn.Module):

    def __init__(self, config, **kwargs):
        super(GodAuxLoss, self).__init__()

        self.hist_seq_len = config.get("hist_seq_len", 0) + 1

        self.loss_tasks = config.get('loss_tasks', ['occupancy', 'semantic'])
        self.loss_weight = config.get('loss_weight', 1.0)

        self.semantic_class = config.get('semantic_class', 7)
        self.decouple_sem = config.get('decouple_sem', False)
        self.god_distill = config.get('god_distill', False)

        if 'occupancy' in self.loss_tasks:
            self.occupancy_weights = config.get('occupancy_weights', [4.0, 4.0, 4.0, 3.0, 2.5, 1.0, 4.0])
            assert len(self.occupancy_weights) == self.semantic_class, 'please check occupancy_weights'
            self.loss_occupancy = nn.BCEWithLogitsLoss(reduction="none")

        if 'semantic' in self.loss_tasks:
            semantic_weights = config.get("semantic_weights", [1.3, 3, 2.0, 1.0, 2.8, 0.05, 3])
            assert len(self.occupancy_weights) == self.semantic_class, 'please check semantic_weights'
            self.loss_semantic = MultiFocalLoss(self.semantic_class, alpha=semantic_weights, gamma=2)

        if 'visibility' in self.loss_tasks:
            self.visibility_weights = config.get("visibility_weights", [1, 1])
            self.loss_visibility = MaskedBCEWithLogitsLoss()

        if 'drivable' in self.loss_tasks:
            self.drivable_weights = config.get('drivable_weights', [1.0, 1.0])
            self.drivable_curb_weight = config.get('drivable_curb_weight', -1.0)
            self.drivable_curb_mask_source = config.get("drivable_curb_mask_source", 'drivable_curb_mask_dilate')
            self.loss_drivable = MaskedBCEWithLogitsLoss()

        if self.god_distill:
            self.loss_distill = nn.MSELoss(reduction='none')

    def forward(self, *inputs, **kwargs):
        inputs = list(inputs)
        uvp_labels = inputs.pop(0)
        labels =  [[]]
        labels[0] = uvp_labels
        pred_dict = inputs[0]['god_aux_head_output']
        # pred_dict = {'distill_feat':torch.zeros((2, 64, 144, 56), device='cuda'),
        # 'occupancy':torch.zeros((2, 16, 288, 112), device='cuda'),
        # 'semantic': torch.zeros((2, 112, 288, 112), device='cuda'),
        # 'visibility': torch.zeros((2, 16, 288, 112), device='cuda'),
        # 'drivable':torch.zeros((2, 1, 288, 112), device='cuda'),
        # }

        gt_dict = labels[0]['pnc_god_head_input']

        occupancy_map_gt = gt_dict['occupancy'] # [2, 16, 288, 112]
        semantic_gt = gt_dict['semantic']       # [2, 16, 288, 112]
        
        bs = semantic_gt.shape[0]
        godseg_valid = torch.ones((bs, 1), device=semantic_gt.device).bool()
        if 'godseg_cache_valid' in labels[0]:
            godseg_valid = godseg_valid & labels[0]['godseg_cache_valid']
        godfeat_valid = torch.ones((bs, 1), device=semantic_gt.device).bool()
        if 'god_feat_cache_valid' in labels[0]:
            godfeat_valid = godfeat_valid & labels[0]['god_feat_cache_valid']

        # counter.count_error('godseg_invalid_frames', (~godseg_valid).sum().item())

        loss_dict = dict()
        if self.decouple_sem:
            unoccupancy_index = (occupancy_map_gt == 0)
            # semantic_gt[unoccupancy_index] = 5
            semantic_gt = semantic_gt * (~unoccupancy_index) + unoccupancy_index * GOD_SEMANTIC_TYPE_MAPPING["UNOCCUPANCY"]

        if 'occupancy' in self.loss_tasks:
            occupancy_map_gt = occupancy_map_gt.float()
            occupancy_loss = self.loss_occupancy(pred_dict['occupancy'], occupancy_map_gt) # [2, 16, 288, 112], [2, 16, 288, 112]

            # different occ weight for different semantic
            occupancy_weights = torch.zeros_like(occupancy_map_gt)
            for s_class in range(self.semantic_class):
                semantic_occ_mask = (semantic_gt == s_class)
                occupancy_weights = occupancy_weights + semantic_occ_mask * self.occupancy_weights[s_class]

            occupancy_loss = occupancy_loss * godseg_valid[..., None, None]
            occupancy_loss = torch.mean(occupancy_loss * occupancy_weights)
            loss_dict['occupancy'] = occupancy_loss

        if 'semantic' in self.loss_tasks:
            default_idx = semantic_gt == -1
            semantic_gt = semantic_gt * (~default_idx) + default_idx

            b, z, h, w = occupancy_map_gt.shape
            semantic_pred = pred_dict['semantic'].view(b, z, self.semantic_class, h, w).permute(0, 2, 1, 3, 4)
            semantic_loss = self.loss_semantic(semantic_pred, semantic_gt)
            semantic_loss = semantic_loss * godseg_valid[..., None, None]
            loss_dict['semantic'] = torch.mean(semantic_loss)

        if 'visibility' in self.loss_tasks:
            visibility_gt = gt_dict['visibility']

            visibility_class_weight = torch.ones_like(visibility_gt) * self.visibility_weights[0]
            visibility_gt_index =(visibility_gt == 1)
            visibility_class_weight = visibility_class_weight * (~visibility_gt_index) + visibility_gt_index * self.visibility_weights[1]

            visibility_loss = self.loss_visibility(pred_dict['visibility'], visibility_gt.float(), weight=visibility_class_weight)
            visibility_loss = visibility_loss * godseg_valid[..., None, None, None]
            loss_dict['visibility'] = torch.mean(visibility_loss)

        if 'drivable' in self.loss_tasks:
            # for drivable area
            drivable_area_gt = gt_dict['drivable']
            drivable_area_gt = drivable_area_gt.float()
            
            drivable_valid_mask = gt_dict.get('drivable_valid_mask', None)
            if drivable_valid_mask is not None:
                drivable_mask = drivable_mask * drivable_valid_mask[..., None]

            drivable_weights = copy.deepcopy(self.drivable_weights)
            drivable_weights = drivable_area_gt * (self.drivable_weights[1] - self.drivable_weights[0]) + self.drivable_weights[0]

            drivable_curb_mask = gt_dict.get(self.drivable_curb_mask_source)
            if drivable_curb_mask is not None and self.drivable_curb_weight > 0:
                # 只对curb区域，提升Loss权重
                drivable_weights = drivable_weights * (drivable_curb_mask[..., None] * (self.drivable_curb_weight - 1) + 1)

            drivable_loss = self.loss_drivable(pred_dict['drivable'], drivable_area_gt, weight=drivable_weights)
            drivable_loss = drivable_loss * godseg_valid[..., None, None]
            loss_dict['drivable'] = torch.mean(drivable_loss)

        if self.god_distill:
            gt_feat = labels[0]['pnc_god_dense_input'] # [2, 64, 144, 56]
            pred_feat = pred_dict['distill_feat']      # [2, 64, 144, 56]
            distill_loss = self.loss_distill(pred_feat, gt_feat)
            distill_loss = distill_loss * godfeat_valid[..., None, None]
            loss_dict['god_distill'] = torch.mean(distill_loss)

        origin_loss = []
        loss_weight = []
        weighted_loss = []
        weighted_loss_dict = {}

        for k, v in loss_dict.items():
            weight = self.loss_weight
            if k == 'god_distill':
                weight = self.loss_weight / 3
            weighted_item = v * weight

            origin_loss.append(v)
            loss_weight.append(weight)
            weighted_loss.append(weighted_item)
            weighted_loss_dict[k] = weighted_item

        final_loss = loss_formatter(
            weighted_loss=weighted_loss,
            origin_loss=origin_loss,
            loss_weight=loss_weight,
        )
        final_loss = _attach_loss_items(final_loss, loss_dict, weighted_loss_dict)

        # try: 
        #     GodHeadWatcher.watch(pred_dict, gt_dict)
        # except Exception as e:
        #     logger.warning(f"{e}")

        return  final_loss
