from typing import Any, Dict, Iterable, List, Mapping, Tuple

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

    def __init__(self, *, intermediate_layer_idx, merge_size: int = 2):
        self.intermediate_layer_idx = list(intermediate_layer_idx)
        self.merge_size = int(merge_size)

    @staticmethod
    def _count_token_runs(input_ids_row: torch.Tensor, token_id: int) -> List[Tuple[int, int]]:
        mask = input_ids_row.eq(token_id)
        if not mask.any():
            return []
        indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        runs = []
        start = indices[0]
        prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            runs.append((start, prev + 1))
            start = idx
            prev = idx
        runs.append((start, prev + 1))
        return runs

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
        if 'input_ids' not in payload:
            raise KeyError('AuxSeparateHeads payload must contain "input_ids".')
        if 'image_grid_thw' not in payload:
            raise KeyError('AuxSeparateHeads payload must contain "image_grid_thw" when using <image> tokens.')
        if 'model_config' not in payload:
            raise KeyError('AuxSeparateHeads payload must contain "model_config".')
        return dict(payload)

    def _collect_image_token_metadata(self, input_ids: torch.Tensor, image_grid_thw: torch.Tensor,
                                      model_config) -> List[Dict[str, int]]:
        image_token_id = getattr(model_config, 'image_token_id', None)
        if image_token_id is None:
            raise ValueError('model_config.image_token_id is required to extract <image> token spans.')

        image_cursor = 0
        metadata = []
        image_grid_thw = image_grid_thw.cpu()
        for sample_idx in range(input_ids.shape[0]):
            runs = self._count_token_runs(input_ids[sample_idx], image_token_id)
            if not runs:
                raise ValueError(f'No <image> token span found for sample {sample_idx}.')
            if image_cursor >= len(image_grid_thw):
                raise ValueError('image_grid_thw is shorter than the number of <image> spans in the batch.')

            start, end = runs[0]
            grid_thw = image_grid_thw[image_cursor]
            image_cursor += 1

            temporal = int(grid_thw[0].item())
            grid_h = int(grid_thw[1].item() // self.merge_size)
            grid_w = int(grid_thw[2].item() // self.merge_size)
            token_count = temporal * grid_h * grid_w
            if (end - start) != token_count:
                raise ValueError(
                    f'<image> token span length mismatch for sample {sample_idx}: '
                    f'run={end - start}, expected={token_count}.')
            if temporal != 1:
                raise ValueError(f'Only image inputs with temporal dimension 1 are supported, got {temporal}.')

            metadata.append({
                'sample_idx': sample_idx,
                'start': start,
                'end': end,
                'grid_h': grid_h,
                'grid_w': grid_w,
                'token_count': token_count,
            })
        return metadata

    @staticmethod
    def _stack_tokens(hidden_states, metadata: List[Dict[str, int]]) -> torch.Tensor:
        token_counts = {item['token_count'] for item in metadata}
        if len(token_counts) != 1:
            raise ValueError(
                f'All samples in the batch must share the same <image> token count. Got: {sorted(token_counts)}')
        tokens = [
            hidden_states[item['sample_idx'], item['start']:item['end'], :].contiguous() for item in metadata
        ]
        return torch.stack(tokens, dim=0)

    def build_hidden_state_payload(self, hidden_states, input_ids: torch.Tensor, image_grid_thw: torch.Tensor, model_config):
        all_tokens_with_pe = hidden_states[-1]
        metadata = self._collect_image_token_metadata(input_ids, image_grid_thw, model_config)
        image_tokens = self._stack_tokens(all_tokens_with_pe, metadata)
        image_tokens_list = []
        for index in self.intermediate_layer_idx:
            image_tokens_list.append(self._stack_tokens(hidden_states[index], metadata))

        grid_h = metadata[0]['grid_h']
        grid_w = metadata[0]['grid_w']

        return {
            'all_tokens_with_pe': all_tokens_with_pe,
            'image_tokens': image_tokens,
            'image_tokens_list': image_tokens_list,
            'image_token_metadata': metadata,
            'grid_h': grid_h,
            'grid_w': grid_w,
        }

    def build_fusion_features(self, hidden_states, input_ids: torch.Tensor, image_grid_thw: torch.Tensor, model_config):
        payload = self.build_hidden_state_payload(hidden_states, input_ids, image_grid_thw, model_config)
        flatten_feat = payload['image_tokens']
        batch_size, _, channel_dim = flatten_feat.shape
        height = payload['grid_h']
        width = payload['grid_w']
        fusion_feat = flatten_feat.reshape(batch_size, height, width, channel_dim).permute(0, 3, 1, 2)
        return payload, fusion_feat, height, width
