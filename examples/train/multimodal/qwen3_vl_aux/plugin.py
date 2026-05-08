import json
import os
from collections import defaultdict
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import safetensors.torch
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer

from swift.llm import deep_getattr, get_multimodal_target_regex
from swift.plugin import Tuner, extra_tuners, loss_mapping, optimizers_map
from swift.plugin.loss import cross_entropy_loss_func
from swift.tuners import LoraConfig, Swift
from swift.utils import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from swift.llm import TrainArguments


AUX_HEAD_CONFIG = 'aux_head_config.json'
AUX_TRAINABLES = 'aux_trainables.safetensors'
AUX_TASKS = ('bev', 'cog', 'god')
AUX_MODULE_KEYWORDS = tuple(
    f'{task}_{suffix}' for task in AUX_TASKS for suffix in ('dpt_head', 'aux_head', 'aux_loss')
)


def _is_true(env_key: str, default: str = 'false') -> bool:
    return os.environ.get(env_key, default).lower() in {'1', 'true', 'y', 'yes'}


def _parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(',') if item.strip()]


def _get_selected_layer_indices() -> List[int]:
    return _parse_int_list(os.environ.get('QWEN3VL_AUX_HIDDEN_LAYERS', '6,13,20,27'))


def _get_merge_size() -> int:
    return int(os.environ.get('QWEN3VL_AUX_MERGE_SIZE', os.environ.get('SPATIAL_MERGE_SIZE', '2')))


def _get_effective_patch_size() -> int:
    return int(os.environ.get('QWEN3VL_AUX_DPT_PATCH_SIZE', '32'))


def _get_dpt_features() -> int:
    return int(os.environ.get('QWEN3VL_AUX_DPT_FEATURES', '256'))


def _get_dpt_out_channels() -> List[int]:
    return _parse_int_list(os.environ.get('QWEN3VL_AUX_DPT_OUT_CHANNELS', '256,512,1024,1024'))


def _get_aux_head_lr(default_lr: float) -> float:
    return float(os.environ.get('QWEN3VL_AUX_HEAD_LR', str(default_lr)))


def _get_lm_weight() -> float:
    return float(os.environ.get('QWEN3VL_AUX_LM_WEIGHT', '1.0'))


def _get_task_num_classes(task: str) -> int:
    return int(os.environ.get(f'QWEN3VL_{task.upper()}_NUM_CLASSES', '2'))


def _get_task_weight(task: str) -> float:
    return float(os.environ.get(f'QWEN3VL_{task.upper()}_LOSS_WEIGHT', '1.0'))


def _parameter_in_prefixes(name: str, prefixes: Sequence[str]) -> bool:
    for prefix in prefixes:
        if name == prefix or name.startswith(f'{prefix}.') or f'.{prefix}.' in name:
            return True
    return False


def _is_aux_parameter_name(name: str) -> bool:
    return any(keyword in name for keyword in AUX_MODULE_KEYWORDS)


def _get_target_model(model: nn.Module) -> nn.Module:
    current = model
    visited = set()
    while isinstance(current, nn.Module) and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, 'lm_head') and hasattr(current, 'config'):
            return current
        next_model = getattr(current, 'model', None)
        if not isinstance(next_model, nn.Module):
            break
        current = next_model
    return model


def _normalize_task_label(value, device: torch.device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.to(device=device)
    elif isinstance(value, (list, tuple)):
        value = torch.as_tensor(value, device=device)
    else:
        value = torch.as_tensor([value], device=device)
    if value.ndim == 0:
        value = value.unsqueeze(0)
    return value.long()


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


def _collect_media_runs(input_ids_row: torch.Tensor, config) -> List[Tuple[str, int, int]]:
    runs = []
    image_token_id = getattr(config, 'image_token_id', None)
    video_token_id = getattr(config, 'video_token_id', None)
    if image_token_id is not None:
        runs += [('image', start, end) for start, end in _count_token_runs(input_ids_row, image_token_id)]
    if video_token_id is not None:
        runs += [('video', start, end) for start, end in _count_token_runs(input_ids_row, video_token_id)]
    return sorted(runs, key=lambda item: item[1])


def _custom_interpolate(x: torch.Tensor,
                        size: Optional[Tuple[int, int]] = None,
                        scale_factor: Optional[float] = None,
                        mode: str = 'bilinear',
                        align_corners: bool = True) -> torch.Tensor:
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class ResidualConvUnit(nn.Module):

    def __init__(self, features: int, groups: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, groups=groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(x, inplace=True)
        out = self.conv1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):

    def __init__(self, features: int, has_residual: bool = True, groups: int = 1):
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.residual = ResidualConvUnit(features, groups=groups)
        self.fuse = ResidualConvUnit(features, groups=groups)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, groups=groups)

    def forward(self,
                x: torch.Tensor,
                residual: Optional[torch.Tensor] = None,
                size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        out = x
        if self.has_residual and residual is not None:
            out = out + self.residual(residual)
        out = self.fuse(out)
        if size is None:
            out = _custom_interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            out = _custom_interpolate(out, size=size, mode='bilinear', align_corners=True)
        return self.out_conv(out)


class Qwen3VLDPTHead(nn.Module):
    """VGGT-style DPT head adapted to Qwen3-VL visual hidden states."""

    def __init__(self,
                 dim_in: int,
                 *,
                 patch_size: int = 32,
                 features: int = 256,
                 out_channels: Optional[List[int]] = None,
                 intermediate_layer_idx: Optional[List[int]] = None,
                 pos_embed: bool = True,
                 feature_only: bool = True,
                 down_ratio: int = 1,
                 merge_size: int = 2) -> None:
        super().__init__()
        out_channels = out_channels or [256, 512, 1024, 1024]
        intermediate_layer_idx = intermediate_layer_idx or [6, 13, 20, 27]
        self.patch_size = patch_size
        self.features = features
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.merge_size = merge_size
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        self.layer_rn = nn.ModuleList(
            [nn.Conv2d(oc, features, kernel_size=3, stride=1, padding=1, bias=False) for oc in out_channels])
        self.refinenet1 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet4 = FeatureFusionBlock(features, has_residual=False)
        self.output_conv = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)

    def _apply_pos_embed(self, x: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        ys = torch.linspace(-1, 1, patch_h, dtype=x.dtype, device=x.device)
        xs = torch.linspace(-1, 1, patch_w, dtype=x.dtype, device=x.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        pos = torch.stack([grid_x, grid_y], dim=0)[None].expand(x.shape[0], -1, -1, -1)
        repeat = (x.shape[1] + pos.shape[1] - 1) // pos.shape[1]
        pos = pos.repeat(1, repeat, 1, 1)[:, :x.shape[1]]
        return x + 0.1 * pos

    def _scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features
        layer_1 = self.layer_rn[0](layer_1)
        layer_2 = self.layer_rn[1](layer_2)
        layer_3 = self.layer_rn[2](layer_3)
        layer_4 = self.layer_rn[3](layer_4)

        out = self.refinenet4(layer_4, size=layer_3.shape[2:])
        out = self.refinenet3(out, layer_3, size=layer_2.shape[2:])
        out = self.refinenet2(out, layer_2, size=layer_1.shape[2:])
        out = self.refinenet1(out, layer_1)
        return self.output_conv(out)

    def forward(self, aggregated_tokens_list: List[torch.Tensor], grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw = grid_thw.to(dtype=torch.long)
        t = int(grid_thw[0].item())
        patch_h = int(grid_thw[1].item() // self.merge_size)
        patch_w = int(grid_thw[2].item() // self.merge_size)
        per_frame_tokens = patch_h * patch_w
        out = []

        for dpt_idx, _ in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[dpt_idx]
            batch_size, num_frames, num_tokens, hidden_size = x.shape
            assert batch_size == 1, f'Expected one media item per forward, got batch_size={batch_size}.'
            if num_frames != t:
                raise ValueError(f'Frame count mismatch in DPT head: expected {t}, got {num_frames}.')
            if num_tokens != per_frame_tokens:
                raise ValueError(
                    f'Visual token count mismatch in DPT head: expected {per_frame_tokens}, got {num_tokens}.')
            x = x.reshape(batch_size * num_frames, num_tokens, hidden_size)
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(batch_size * num_frames, hidden_size, patch_h, patch_w)
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, patch_h, patch_w)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)

        out = self._scratch_forward(out)
        out = _custom_interpolate(
            out,
            size=(int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode='bilinear',
            align_corners=True)
        if self.pos_embed:
            out = self._apply_pos_embed(out, out.shape[-2], out.shape[-1])
        if self.feature_only:
            return out.view(1, t, *out.shape[1:])
        return out


class BaseAuxHead(nn.Module):

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(in_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [1, S, C, H, W]
        x = features.mean(dim=1)
        x = self.conv(x).flatten(1)
        return self.classifier(x)


class BevAuxHead(BaseAuxHead):
    pass


class CogAuxHead(BaseAuxHead):
    pass


class GodAuxHead(BaseAuxHead):
    pass


class BaseAuxLoss(nn.Module):

    def forward(self, logits: torch.Tensor, targets: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if targets is None:
            return None
        return F.cross_entropy(logits, targets)


class BevAuxLoss(BaseAuxLoss):
    pass


class CogAuxLoss(BaseAuxLoss):
    pass


class GodAuxLoss(BaseAuxLoss):
    pass


def _build_aux_config(target_model: nn.Module) -> Dict[str, object]:
    text_config = getattr(target_model.config, 'text_config', target_model.config)
    hidden_size = getattr(text_config, 'hidden_size', None) or getattr(target_model.config, 'hidden_size')
    return {
        'layer_indices': _get_selected_layer_indices(),
        'hidden_size': hidden_size,
        'merge_size': _get_merge_size(),
        'patch_size': _get_effective_patch_size(),
        'features': _get_dpt_features(),
        'out_channels': _get_dpt_out_channels(),
        'task_num_classes': {task: _get_task_num_classes(task) for task in AUX_TASKS},
    }


def _attach_task_modules(target_model: nn.Module, aux_config: Dict[str, object]) -> None:
    hidden_size = int(aux_config['hidden_size'])
    dpt_kwargs = {
        'dim_in': hidden_size,
        'patch_size': int(aux_config['patch_size']),
        'features': int(aux_config['features']),
        'out_channels': list(aux_config['out_channels']),
        'intermediate_layer_idx': list(aux_config['layer_indices']),
        'merge_size': int(aux_config['merge_size']),
        'feature_only': True,
    }
    num_classes = aux_config['task_num_classes']
    target_model.bev_dpt_head = Qwen3VLDPTHead(**dpt_kwargs)
    target_model.cog_dpt_head = Qwen3VLDPTHead(**dpt_kwargs)
    target_model.god_dpt_head = Qwen3VLDPTHead(**dpt_kwargs)
    target_model.bev_aux_head = BevAuxHead(in_channels=int(aux_config['features']), num_classes=num_classes['bev'])
    target_model.cog_aux_head = CogAuxHead(in_channels=int(aux_config['features']), num_classes=num_classes['cog'])
    target_model.god_aux_head = GodAuxHead(in_channels=int(aux_config['features']), num_classes=num_classes['god'])
    target_model.bev_aux_loss = BevAuxLoss()
    target_model.cog_aux_loss = CogAuxLoss()
    target_model.god_aux_loss = GodAuxLoss()
    target_model.aux_head_config = aux_config


def _extract_selected_hidden_states(outputs, layer_indices: Sequence[int]) -> List[torch.Tensor]:
    hidden_states = getattr(outputs, 'hidden_states', None) or []
    if not hidden_states:
        return []
    selected = []
    for idx in layer_indices:
        if idx < len(hidden_states):
            selected.append(hidden_states[idx])
        else:
            logger.warning(f'Skip hidden state index {idx}; only {len(hidden_states)} hidden states are available.')
    return selected


def _build_media_entries(selected_hidden_states: List[torch.Tensor],
                         input_ids: torch.Tensor,
                         config,
                         image_grid_thw: Optional[torch.Tensor],
                         video_grid_thw: Optional[torch.Tensor],
                         merge_size: int) -> List[Dict[str, object]]:
    if not selected_hidden_states:
        return []
    image_grid_thw = image_grid_thw.cpu() if image_grid_thw is not None else None
    video_grid_thw = video_grid_thw.cpu() if video_grid_thw is not None else None
    image_cursor = 0
    video_cursor = 0
    media_entries: List[Dict[str, object]] = []

    for sample_idx in range(input_ids.shape[0]):
        runs = _collect_media_runs(input_ids[sample_idx], config)
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
            expected_tokens = t * per_frame_tokens
            if expected_tokens != token_count:
                logger.warning(
                    f'Skip one {media_type} item because token count mismatch: expected {expected_tokens}, '
                    f'got {token_count}. sample_idx={sample_idx}')
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


def _aggregate_logits(per_sample_logits: Dict[int, List[torch.Tensor]], batch_size: int, num_classes: int,
                      device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    aggregated = []
    zero = torch.zeros(1, num_classes, device=device, dtype=dtype)
    for sample_idx in range(batch_size):
        logits_list = per_sample_logits.get(sample_idx, None)
        if logits_list:
            aggregated.append(torch.stack(logits_list, dim=0).mean(dim=0))
        else:
            aggregated.append(zero.clone())
    return torch.cat(aggregated, dim=0)


def _run_task_branch(target_model: nn.Module,
                     task: str,
                     media_entries: List[Dict[str, object]],
                     batch_size: int,
                     device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
    dpt_head = getattr(target_model, f'{task}_dpt_head')
    aux_head = getattr(target_model, f'{task}_aux_head')
    num_classes = aux_head.num_classes
    per_sample_logits: Dict[int, List[torch.Tensor]] = defaultdict(list)
    for entry in media_entries:
        feature_maps = dpt_head(entry['hidden_states'], entry['grid_thw'])
        logits = aux_head(feature_maps)
        per_sample_logits[entry['sample_idx']].append(logits)
    return _aggregate_logits(per_sample_logits, batch_size, num_classes, device, dtype)


def attach_auxiliary_modules(model: nn.Module, aux_config: Optional[Dict[str, object]] = None) -> nn.Module:
    target_model = _get_target_model(model)
    if getattr(target_model, '_qwen3vl_aux_patched', False):
        return model

    aux_config = aux_config or _build_aux_config(target_model)
    _attach_task_modules(target_model, aux_config)
    target_model.to(device=next(target_model.parameters()).device, dtype=next(target_model.parameters()).dtype)

    origin_forward = target_model.forward

    def forward(self,
                *args,
                y_bev=None,
                y_cog=None,
                y_god=None,
                **kwargs):
        kwargs.setdefault('return_dict', True)
        kwargs['output_hidden_states'] = True
        input_ids = kwargs.get('input_ids')
        attention_mask = kwargs.get('attention_mask')
        image_grid_thw = kwargs.get('image_grid_thw')
        video_grid_thw = kwargs.get('video_grid_thw')
        if input_ids is None and args:
            input_ids = args[0]

        outputs = origin_forward(*args, **kwargs)
        outputs.bev_logits = None
        outputs.cog_logits = None
        outputs.god_logits = None
        outputs.bev_aux_loss = None
        outputs.cog_aux_loss = None
        outputs.god_aux_loss = None
        outputs.aux_loss = None

        if input_ids is None:
            return outputs

        selected_hidden_states = _extract_selected_hidden_states(outputs, self.aux_head_config['layer_indices'])
        if not selected_hidden_states:
            return outputs

        media_entries = _build_media_entries(
            selected_hidden_states,
            input_ids,
            self.config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            merge_size=int(self.aux_head_config['merge_size']))
        if not media_entries:
            return outputs

        batch_size = input_ids.shape[0]
        dtype = selected_hidden_states[0].dtype
        device = selected_hidden_states[0].device
        outputs.visual_hidden_state_layers = self.aux_head_config['layer_indices']

        outputs.bev_logits = _run_task_branch(self, 'bev', media_entries, batch_size, device, dtype)
        outputs.cog_logits = _run_task_branch(self, 'cog', media_entries, batch_size, device, dtype)
        outputs.god_logits = _run_task_branch(self, 'god', media_entries, batch_size, device, dtype)

        y_bev = _normalize_task_label(y_bev, device)
        y_cog = _normalize_task_label(y_cog, device)
        y_god = _normalize_task_label(y_god, device)

        outputs.bev_aux_loss = self.bev_aux_loss(outputs.bev_logits, y_bev)
        outputs.cog_aux_loss = self.cog_aux_loss(outputs.cog_logits, y_cog)
        outputs.god_aux_loss = self.god_aux_loss(outputs.god_logits, y_god)

        aux_losses = [loss for loss in (outputs.bev_aux_loss, outputs.cog_aux_loss, outputs.god_aux_loss) if loss is not None]
        if aux_losses:
            outputs.aux_loss = torch.stack(aux_losses).sum()
        return outputs

    target_model.forward = MethodType(forward, target_model)
    target_model._qwen3vl_aux_patched = True
    logger.info(f'Attached Qwen3-VL multi-task auxiliary branches with config: {aux_config}')
    return model


def qwen3vl_aux_loss(outputs, labels, num_items_in_batch=None, trainer=None, **kwargs) -> torch.Tensor:
    lm_loss = cross_entropy_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **kwargs)
    loss = _get_lm_weight() * lm_loss
    task_losses = {
        'bev': getattr(outputs, 'bev_aux_loss', None),
        'cog': getattr(outputs, 'cog_aux_loss', None),
        'god': getattr(outputs, 'god_aux_loss', None),
    }
    if trainer is not None:
        mode = 'train' if trainer.model.training else 'eval'
        trainer.custom_metrics[mode]['lm_loss'].update(lm_loss.detach())
    for task, task_loss in task_losses.items():
        if task_loss is None:
            continue
        if trainer is not None:
            trainer.custom_metrics[mode][f'{task}_aux_loss'].update(task_loss.detach())
        loss = loss + _get_task_weight(task) * task_loss
    return loss


class Qwen3VLAuxTuner(Tuner):

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        model_arch = model.model_meta.model_arch
        target_regex = get_multimodal_target_regex(model)
        logger.info(f'target_regex: {target_regex}')
        lora_config = LoraConfig(
            task_type='CAUSAL_LM', r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=target_regex)
        model = Swift.prepare_model(model, lora_config)
        model = attach_auxiliary_modules(model)

        if _is_true('QWEN3VL_AUX_TRAIN_VISION') or args.vit_lr is not None or args.aligner_lr is not None:
            for module_prefix in model_arch.vision_tower + model_arch.aligner:
                deep_getattr(model, module_prefix).requires_grad_(True)
            logger.info('Enabled full-parameter training for vision tower / aligner.')
        return model

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        if state_dict is None:
            state_dict = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        model.save_pretrained(save_directory, state_dict=state_dict, safe_serialization=safe_serialization, **kwargs)

        target_model = _get_target_model(model)
        aux_config = getattr(target_model, 'aux_head_config', _build_aux_config(target_model))
        with open(os.path.join(save_directory, AUX_HEAD_CONFIG), 'w', encoding='utf-8') as f:
            json.dump(aux_config, f, ensure_ascii=False, indent=2)

        aux_state_dict = {}
        for name, value in state_dict.items():
            if _is_aux_parameter_name(name):
                aux_state_dict[name] = value
        if aux_state_dict:
            safetensors.torch.save_file(
                aux_state_dict, os.path.join(save_directory, AUX_TRAINABLES), metadata={'format': 'pt'})

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        model = Swift.from_pretrained(model, model_id, **kwargs)
        aux_config_path = os.path.join(model_id, AUX_HEAD_CONFIG)
        aux_config = None
        if os.path.exists(aux_config_path):
            with open(aux_config_path, 'r', encoding='utf-8') as f:
                aux_config = json.load(f)
        model = attach_auxiliary_modules(model, aux_config=aux_config)

        aux_weights_path = os.path.join(model_id, AUX_TRAINABLES)
        if os.path.exists(aux_weights_path):
            state_dict = safetensors.torch.load_file(aux_weights_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(f'Loaded auxiliary trainables from {aux_weights_path}.')
            if missing:
                logger.warning(f'Missing keys when loading aux trainables: {missing}')
            if unexpected:
                logger.warning(f'Unexpected keys when loading aux trainables: {unexpected}')
        return model


def create_qwen3vl_aux_optimizer(args: 'TrainArguments', model, dataset):
    decay_parameters = set(Trainer.get_decay_parameter_names(None, model))
    model_arch = model.model_meta.model_arch
    aux_head_lr = _get_aux_head_lr(args.learning_rate)
    vit_lr = args.vit_lr if args.vit_lr is not None else args.learning_rate
    aligner_lr = args.aligner_lr if args.aligner_lr is not None else args.learning_rate
    logger.info(
        f'aux_head_lr: {aux_head_lr}, vit_lr: {vit_lr}, aligner_lr: {aligner_lr}, llm_lr: {args.learning_rate}')

    groups = []
    seen = set()
    named_parameters = list(model.named_parameters())

    def add_group(lr: float, predicate):
        for use_wd, weight_decay in ((False, 0.), (True, args.weight_decay)):
            params = []
            for name, param in named_parameters:
                if not param.requires_grad or id(param) in seen or not predicate(name):
                    continue
                if use_wd != (name in decay_parameters):
                    continue
                params.append(param)
                seen.add(id(param))
            if params:
                groups.append({'params': params, 'weight_decay': weight_decay, 'lr': lr})

    add_group(aux_head_lr, _is_aux_parameter_name)
    add_group(vit_lr, lambda name: _parameter_in_prefixes(name, model_arch.vision_tower))
    add_group(aligner_lr, lambda name: _parameter_in_prefixes(name, model_arch.aligner))
    add_group(args.learning_rate, lambda name: True)

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    return optimizer_cls(groups, **optimizer_kwargs), None


extra_tuners['qwen3vl_aux'] = Qwen3VLAuxTuner
loss_mapping['qwen3vl_aux'] = qwen3vl_aux_loss
optimizers_map['qwen3vl_aux'] = create_qwen3vl_aux_optimizer
