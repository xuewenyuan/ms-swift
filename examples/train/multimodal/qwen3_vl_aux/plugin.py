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
from common.label_loader import Qwen3VLAuxLabelLoader
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


def _get_active_tasks(aux_config: Dict[str, object]) -> Sequence[str]:
    tasks_cfg = aux_config.get('tasks', {})
    return [task for task in AUX_TASKS if tasks_cfg.get(task, {}).get('enabled', True)]


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


def _extract_loss_tensor(loss_output: Any, task: str) -> Optional[torch.Tensor]:
    def _sum_tensors(value: Any) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, Mapping):
            total = None
            for nested in value.values():
                tensor = _sum_tensors(nested)
                if tensor is None:
                    continue
                total = tensor if total is None else total + tensor
            return total
        if isinstance(value, (list, tuple)):
            total = None
            for nested in value:
                tensor = _sum_tensors(nested)
                if tensor is None:
                    continue
                total = tensor if total is None else total + tensor
            return total
        return None

    if loss_output is None:
        return None
    if isinstance(loss_output, torch.Tensor):
        return loss_output
    if isinstance(loss_output, Mapping):
        loss = loss_output.get('loss')
        if isinstance(loss, torch.Tensor):
            return loss
        for key in ('weighted_loss', 'total_loss', 'final_loss', 'origin_loss'):
            tensor = _sum_tensors(loss_output.get(key))
            if tensor is not None:
                return tensor
        raise TypeError(f'Loss mapping for task "{task}" must contain a tensor field named "loss".')
    loss = getattr(loss_output, 'loss', None)
    if isinstance(loss, torch.Tensor):
        return loss
    for attr_name in ('weighted_loss', 'total_loss', 'final_loss', 'origin_loss'):
        tensor = _sum_tensors(getattr(loss_output, attr_name, None))
        if tensor is not None:
            return tensor
    raise TypeError(f'Unsupported auxiliary loss output type for task "{task}": {type(loss_output)!r}.')


def _collect_task_losses_from_outputs(outputs: Any) -> Dict[str, torch.Tensor]:
    task_losses = dict(getattr(outputs, 'aux_task_losses', None) or {})
    active_tasks = getattr(outputs, 'aux_enabled_tasks', None) or AUX_TASKS
    for task in active_tasks:
        if task in task_losses and task_losses[task] is not None:
            continue
        attr_loss = getattr(outputs, f'{task}_aux_loss', None)
        if isinstance(attr_loss, torch.Tensor):
            task_losses[task] = attr_loss
    return {task: loss for task, loss in task_losses.items() if isinstance(loss, torch.Tensor)}


def _pop_label_path_inputs(kwargs: Dict[str, object], preferred_key: str) -> Dict[str, object]:
    label_inputs = {}
    candidate_keys = [preferred_key, 'aux_label_path', 'labels_path', 'label_path', 'src_label_path', 'labels_file']
    for key in candidate_keys:
        if key in kwargs:
            label_inputs[preferred_key] = kwargs.pop(key)
            break
    return label_inputs


def _attach_task_modules(target_model: nn.Module, aux_config: Dict[str, object]) -> None:
    target_model.aux_label_loader = Qwen3VLAuxLabelLoader(aux_config)
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
        active_tasks = _get_active_tasks(self.aux_head_config)
        extra_task_targets = {}
        path_key = self.aux_head_config['shared'].get('label_loader', {}).get('path_key', 'labels_path')
        extra_label_inputs = _pop_label_path_inputs(kwargs, path_key)
        for task in active_tasks:
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
        self._qwen3vl_last_aux_task_losses = {}
        self._qwen3vl_last_aux_total_loss = None
        self._qwen3vl_last_aux_loss_weights = {}
        self._qwen3vl_last_aux_enabled_tasks = list(active_tasks)

        if input_ids is None:
            return outputs

        hidden_states = getattr(outputs, 'hidden_states', None)
        if not hidden_states:
            return outputs
        image_grid_thw = kwargs.get('image_grid_thw')
        if image_grid_thw is None:
            return outputs

        outputs.aux_features = {'hidden_states': hidden_states}
        outputs.visual_hidden_state_layers = list(self.aux_head_config['shared'].get('layer_indices', []))
        outputs.aux_enabled_tasks = list(active_tasks)
        outputs.aux_loss_weights = {
            task: self.aux_head_config['tasks'][task]['loss_weight'] for task in active_tasks
        }
        self._qwen3vl_last_aux_loss_weights = dict(outputs.aux_loss_weights)
        raw_targets = {'bev': y_bev, 'cog': y_cog, 'god': y_god}
        raw_targets.update(extra_task_targets)
        aux_labels = self.aux_label_loader(
            batch_size=input_ids.shape[0],
            batch_kwargs=extra_label_inputs,
            raw_targets=raw_targets,
            device=hidden_states[-1].device,
        )
        outputs.aux_labels = aux_labels

        try:
            head_outputs = self.aux_heads(
                aux_labels.get('batched_shared_labels'),
                {
                    'hidden_states': hidden_states,
                    'input_ids': input_ids,
                    'image_grid_thw': image_grid_thw,
                    'model_config': self.config,
                },
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                'The auxiliary head stack is still an interface stub. '
                'Please implement the required head or upsampler components.') from exc
        if isinstance(head_outputs, list):
            if not head_outputs:
                return outputs
            head_outputs = head_outputs[0]
        if not isinstance(head_outputs, Mapping):
            raise TypeError(f'AuxSeparateHeads must return a mapping or a single-item list, got {type(head_outputs)!r}.')
        outputs.aux_head_outputs = head_outputs

        for task in active_tasks:
            prediction = head_outputs.get(f'{task}_aux_head_output')
            if prediction is None:
                continue
            outputs.aux_predictions[task] = prediction
            setattr(outputs, f'{task}_prediction', prediction)

            try:
                loss_output = self.aux_losses(
                    task,
                    predictions=prediction,
                    targets=aux_labels['task_targets'][task],
                    aux_labels=aux_labels,
                    aux_features=outputs.aux_features,
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
            self._qwen3vl_last_aux_task_losses[task] = task_loss

        loss_values = [loss for loss in outputs.aux_task_losses.values() if loss is not None]
        if loss_values:
            total = loss_values[0]
            for extra in loss_values[1:]:
                total = total + extra
            outputs.aux_loss = total
            self._qwen3vl_last_aux_total_loss = total
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
    task_losses = _collect_task_losses_from_outputs(outputs)
    task_weights = getattr(outputs, 'aux_loss_weights', None) or {}
    mode = None
    target_model = None
    if trainer is not None:
        mode = 'train' if trainer.model.training else 'eval'
        target_model = _get_target_model(trainer.model)
        trainer.custom_metrics[mode]['lm_loss'].update(lm_loss.detach())
    if not task_losses and target_model is not None:
        task_losses = dict(getattr(target_model, '_qwen3vl_last_aux_task_losses', None) or {})
    if not task_weights and target_model is not None:
        task_weights = dict(getattr(target_model, '_qwen3vl_last_aux_loss_weights', None) or {})
    aux_total = None
    for task, task_loss in task_losses.items():
        if task_loss is None:
            continue
        if trainer is not None:
            trainer.custom_metrics[mode][f'{task}_aux_loss'].update(task_loss.detach())
        aux_total = task_loss if aux_total is None else aux_total + task_loss
        loss = loss + float(task_weights.get(task, 1.0)) * task_loss
    if aux_total is None:
        aux_total = getattr(outputs, 'aux_loss', None)
    if aux_total is None and target_model is not None:
        aux_total = getattr(target_model, '_qwen3vl_last_aux_total_loss', None)
    if trainer is not None and isinstance(aux_total, torch.Tensor):
        trainer.custom_metrics[mode]['aux_loss'].update(aux_total.detach())
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
