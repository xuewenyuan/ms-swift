import json
import os
import sys
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

import safetensors.torch
import torch
from torch import nn
from transformers import Trainer

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

from aux_heads import AuxSeparateHeads
from aux_losses import AuxSeparateLosses
from common.config import AUX_TASKS, build_aux_config, get_aux_head_lr
from common.visual_feature_extractor import Qwen3VLAuxFeatureExtractor
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
AUX_MODULE_KEYWORDS = ('aux_heads', 'aux_losses')


def _is_true(env_key: str, default: str = 'false') -> bool:
    return os.environ.get(env_key, default).lower() in {'1', 'true', 'y', 'yes'}


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


def _normalize_task_target(value: Any, device: torch.device) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _normalize_task_target(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_normalize_task_target(item, device) for item in value)
    if isinstance(value, list):
        try:
            return torch.as_tensor(value, device=device)
        except (TypeError, ValueError):
            return [_normalize_task_target(item, device) for item in value]
    try:
        return torch.as_tensor(value, device=device)
    except (TypeError, ValueError):
        return value


def _extract_loss_tensor(loss_output: Any, task: str) -> Optional[torch.Tensor]:
    if loss_output is None:
        return None
    if isinstance(loss_output, torch.Tensor):
        return loss_output
    if isinstance(loss_output, Mapping):
        loss = loss_output.get('loss')
        if isinstance(loss, torch.Tensor):
            return loss
        raise TypeError(f'Loss mapping for task "{task}" must contain a tensor field named "loss".')
    loss = getattr(loss_output, 'loss', None)
    if isinstance(loss, torch.Tensor):
        return loss
    raise TypeError(f'Unsupported auxiliary loss output type for task "{task}": {type(loss_output)!r}.')


def _attach_task_modules(target_model: nn.Module, aux_config: Dict[str, object]) -> None:
    target_model.aux_feature_extractor = Qwen3VLAuxFeatureExtractor(aux_config)
    target_model.aux_heads = AuxSeparateHeads(aux_config)
    target_model.aux_losses = AuxSeparateLosses(aux_config)
    target_model.aux_head_config = aux_config


def attach_auxiliary_modules(model: nn.Module, aux_config: Optional[Dict[str, object]] = None) -> nn.Module:
    target_model = _get_target_model(model)
    if getattr(target_model, '_qwen3vl_aux_patched', False):
        return model

    aux_config = aux_config or build_aux_config(target_model)
    _attach_task_modules(target_model, aux_config)
    target_model.to(device=next(target_model.parameters()).device, dtype=next(target_model.parameters()).dtype)
    origin_forward = target_model.forward

    def forward(self, *args, y_bev=None, y_cog=None, y_god=None, **kwargs):
        extra_task_targets = {}
        for task in AUX_TASKS:
            label_key = self.aux_head_config['tasks'][task]['label_key']
            if label_key not in {'y_bev', 'y_cog', 'y_god'} and label_key in kwargs:
                extra_task_targets[task] = kwargs.pop(label_key)
        kwargs.setdefault('return_dict', True)
        kwargs['output_hidden_states'] = True
        input_ids = kwargs.get('input_ids')
        if input_ids is None and args:
            input_ids = args[0]

        outputs = origin_forward(*args, **kwargs)
        outputs.aux_plugin_config = self.aux_head_config
        outputs.aux_predictions = {}
        outputs.aux_task_losses = {}
        outputs.aux_loss = None

        if input_ids is None:
            return outputs

        aux_features = self.aux_feature_extractor(
            model_outputs=outputs,
            input_ids=input_ids,
            model_config=self.config,
            image_grid_thw=kwargs.get('image_grid_thw'),
            video_grid_thw=kwargs.get('video_grid_thw'),
        )
        if aux_features is None:
            return outputs

        outputs.aux_features = aux_features
        outputs.visual_hidden_state_layers = aux_features['layer_indices']
        outputs.aux_loss_weights = {
            task: self.aux_head_config['tasks'][task]['loss_weight'] for task in AUX_TASKS
        }
        raw_targets = {'bev': y_bev, 'cog': y_cog, 'god': y_god}
        raw_targets.update(extra_task_targets)
        targets = {
            task: _normalize_task_target(raw_targets[task], aux_features['device']) for task in AUX_TASKS
        }

        for task in AUX_TASKS:
            try:
                prediction = self.aux_heads(
                    task,
                    aux_features=aux_features,
                    model_outputs=outputs,
                    task_cfg=self.aux_head_config['tasks'][task],
                    shared_cfg=self.aux_head_config['shared'],
                )
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f'The auxiliary head for task "{task}" is still an interface stub. '
                    f'Please implement {task.title()}AuxHead.forward(...).') from exc
            outputs.aux_predictions[task] = prediction
            setattr(outputs, f'{task}_prediction', prediction)

            try:
                loss_output = self.aux_losses(
                    task,
                    predictions=prediction,
                    targets=targets[task],
                    aux_features=aux_features,
                    model_outputs=outputs,
                    task_cfg=self.aux_head_config['tasks'][task],
                    shared_cfg=self.aux_head_config['shared'],
                )
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f'The auxiliary loss for task "{task}" is still an interface stub. '
                    f'Please implement {task.title()}AuxLoss.forward(...).') from exc
            task_loss = _extract_loss_tensor(loss_output, task)
            outputs.aux_task_losses[task] = task_loss
            setattr(outputs, f'{task}_aux_loss', task_loss)

        loss_values = [loss for loss in outputs.aux_task_losses.values() if loss is not None]
        if loss_values:
            total = loss_values[0]
            for extra in loss_values[1:]:
                total = total + extra
            outputs.aux_loss = total
        return outputs

    target_model.forward = MethodType(forward, target_model)
    target_model._qwen3vl_aux_patched = True
    logger.info(f'Attached Qwen3-VL auxiliary plugin with config: {aux_config}')
    return model


def qwen3vl_aux_loss(outputs, labels, num_items_in_batch=None, trainer=None, **kwargs) -> torch.Tensor:
    lm_loss = cross_entropy_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **kwargs)
    aux_cfg = getattr(outputs, 'aux_plugin_config', None)
    lm_loss_weight = 1.0
    if aux_cfg is not None:
        lm_loss_weight = float(aux_cfg['shared'].get('lm_loss_weight', 1.0))
    loss = lm_loss_weight * lm_loss
    task_losses = getattr(outputs, 'aux_task_losses', None) or {}
    task_weights = getattr(outputs, 'aux_loss_weights', None) or {}
    mode = None
    if trainer is not None:
        mode = 'train' if trainer.model.training else 'eval'
        trainer.custom_metrics[mode]['lm_loss'].update(lm_loss.detach())
    for task, task_loss in task_losses.items():
        if task_loss is None:
            continue
        if trainer is not None:
            trainer.custom_metrics[mode][f'{task}_aux_loss'].update(task_loss.detach())
        loss = loss + float(task_weights.get(task, 1.0)) * task_loss
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
        aux_config = getattr(target_model, 'aux_head_config', build_aux_config(target_model))
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
    aux_head_lr = get_aux_head_lr(args.learning_rate)
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
