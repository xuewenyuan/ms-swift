from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


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


def _collect_media_runs(input_ids_row: torch.Tensor, model_config) -> List[Tuple[str, int, int]]:
    runs = []
    image_token_id = getattr(model_config, 'image_token_id', None)
    video_token_id = getattr(model_config, 'video_token_id', None)
    if image_token_id is not None:
        runs += [('image', start, end) for start, end in _count_token_runs(input_ids_row, image_token_id)]
    if video_token_id is not None:
        runs += [('video', start, end) for start, end in _count_token_runs(input_ids_row, video_token_id)]
    return sorted(runs, key=lambda item: item[1])


class Qwen3VLAuxFeatureExtractor(nn.Module):

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg

    def _extract_selected_hidden_states(self, model_outputs) -> List[torch.Tensor]:
        hidden_states = getattr(model_outputs, 'hidden_states', None) or []
        if not hidden_states:
            return []
        selected = []
        for idx in self.cfg['shared']['layer_indices']:
            if idx < len(hidden_states):
                selected.append(hidden_states[idx])
        return selected

    def _build_media_entries(self,
                             selected_hidden_states: List[torch.Tensor],
                             input_ids: torch.Tensor,
                             model_config,
                             image_grid_thw: Optional[torch.Tensor],
                             video_grid_thw: Optional[torch.Tensor]) -> List[Dict]:
        if not selected_hidden_states:
            return []
        merge_size = int(self.cfg['shared']['merge_size'])
        image_grid_thw = image_grid_thw.cpu() if image_grid_thw is not None else None
        video_grid_thw = video_grid_thw.cpu() if video_grid_thw is not None else None
        image_cursor = 0
        video_cursor = 0
        media_entries: List[Dict] = []

        for sample_idx in range(input_ids.shape[0]):
            runs = _collect_media_runs(input_ids[sample_idx], model_config)
            for media_type, start, end in runs:
                if media_type == 'image':
                    if image_grid_thw is None or image_cursor >= len(image_grid_thw):
                        continue
                    grid_thw = image_grid_thw[image_cursor]
                    image_cursor += 1
                else:
                    if video_grid_thw is None or video_cursor >= len(video_grid_thw):
                        continue
                    grid_thw = video_grid_thw[video_cursor]
                    video_cursor += 1

                t = int(grid_thw[0].item())
                patch_h = int(grid_thw[1].item() // merge_size)
                patch_w = int(grid_thw[2].item() // merge_size)
                per_frame_tokens = patch_h * patch_w
                token_count = end - start
                if t * per_frame_tokens != token_count:
                    continue

                per_layer_tokens = []
                for hidden_states in selected_hidden_states:
                    media_tokens = hidden_states[sample_idx, start:end, :].contiguous()
                    media_tokens = media_tokens.view(t, per_frame_tokens, media_tokens.shape[-1]).unsqueeze(0)
                    per_layer_tokens.append(media_tokens)

                media_entries.append({
                    'sample_idx': sample_idx,
                    'media_type': media_type,
                    'grid_thw': grid_thw.to(device=input_ids.device),
                    'hidden_states': per_layer_tokens,
                })
        return media_entries

    def forward(self,
                model_outputs,
                input_ids: torch.Tensor,
                model_config,
                image_grid_thw: Optional[torch.Tensor] = None,
                video_grid_thw: Optional[torch.Tensor] = None):
        selected_hidden_states = self._extract_selected_hidden_states(model_outputs)
        if not selected_hidden_states:
            return None
        media_entries = self._build_media_entries(
            selected_hidden_states,
            input_ids,
            model_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw)
        if not media_entries:
            return None
        return {
            'layer_indices': list(self.cfg['shared']['layer_indices']),
            'batch_size': int(input_ids.shape[0]),
            'device': selected_hidden_states[0].device,
            'dtype': selected_hidden_states[0].dtype,
            'media_entries': media_entries,
            'shared_cfg': self.cfg['shared'],
        }
