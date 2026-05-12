from typing import Any, Dict, Iterable, Mapping

import torch


def crop_bev_area_a2b(area_a, area_b, resolution):
    crop_area_lft = area_a[1][1] - area_b[1][1]
    crop_area_rht = crop_area_lft + area_b[1][1] - area_b[1][0]

    crop_area_top = area_a[0][1] - area_b[0][1]
    crop_area_down = crop_area_top + area_b[0][1] - area_b[0][0]

    seg_crop_lft = round(crop_area_lft / resolution)
    seg_crop_rht = round(crop_area_rht / resolution)
    seg_crop_top = round(crop_area_top / resolution)
    seg_crop_down = round(crop_area_down / resolution)

    return seg_crop_lft, seg_crop_rht, seg_crop_top, seg_crop_down


class Qwen3VLAuxTokenFeatureAdapter:

    def __init__(self, *, bev_area, foundation_resolution: float, pv_token_len: int, bev_token_len: int,
                 navi_token_len: int, intermediate_layer_idx):
        self.bev_area = bev_area
        self.foundation_resolution = foundation_resolution
        self.pv_token_len = pv_token_len
        self.bev_token_len = bev_token_len
        self.navi_token_len = navi_token_len
        self.intermediate_layer_idx = list(intermediate_layer_idx)

    @staticmethod
    def normalize_head_inputs(inputs: Iterable[Any]) -> Dict[str, Any]:
        items = list(inputs)
        if not items:
            raise ValueError('AuxSeparateHeads expects the hidden-state payload as the last positional argument.')
        payload = items[-1]
        if not isinstance(payload, Mapping):
            raise TypeError(f'AuxSeparateHeads expects a mapping payload, but got {type(payload)!r}.')
        if 'hidden_states' not in payload:
            raise KeyError('AuxSeparateHeads payload must contain "hidden_states".')
        return dict(payload)

    def build_hidden_state_payload(self, hidden_states):
        all_tokens_with_pe = hidden_states[-1]
        batch_size, _, channel_dim = all_tokens_with_pe.shape
        test_token_len = all_tokens_with_pe.shape[1] - (self.pv_token_len + self.bev_token_len + self.navi_token_len)

        pv_tokens, bev_tokens, navi_tokens, test_traj_tokens = torch.split(
            all_tokens_with_pe, [self.pv_token_len, self.bev_token_len, self.navi_token_len, test_token_len], dim=1)

        bev_tokens_list = []
        pv_front_tokens_list = []
        pv_rear_side_tokens_list = []
        for index in self.intermediate_layer_idx:
            layer_hidden = hidden_states[index]
            bev_tokens_list.append(layer_hidden[:, self.pv_token_len:self.pv_token_len + self.bev_token_len])
            pv_front_tokens_list.append([layer_hidden[:, :440].view(batch_size, -1, 2, channel_dim)])
            pv_rear_side_tokens_list.append([layer_hidden[:, 440:self.pv_token_len].view(batch_size, -1, 5, channel_dim)])

        return {
            'all_tokens_with_pe': all_tokens_with_pe,
            'pv_tokens': pv_tokens,
            'bev_tokens': bev_tokens,
            'navi_tokens': navi_tokens,
            'test_traj_tokens': test_traj_tokens,
            'bev_tokens_list': bev_tokens_list,
            'pv_front_tokens_list': pv_front_tokens_list,
            'pv_rear_side_tokens_list': pv_rear_side_tokens_list,
        }

    def build_fusion_features(self, hidden_states):
        payload = self.build_hidden_state_payload(hidden_states)
        flatten_feat = payload['bev_tokens']
        batch_size, _, channel_dim = flatten_feat.shape
        height = round((self.bev_area[0][1] - self.bev_area[0][0]) / self.foundation_resolution)
        width = round((self.bev_area[1][1] - self.bev_area[1][0]) / self.foundation_resolution)
        fusion_feat = flatten_feat.reshape(batch_size, height, width, channel_dim).permute(0, 3, 1, 2)
        return payload, fusion_feat, height, width
