import json
import os
import sys
from numbers import Number
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
from common.plugin_runtime import (AUX_STATE_ATTR, collect_task_losses_from_outputs, get_aux_state, get_target_model,
                                   get_value, make_aux_state, set_aux_state, set_value)
from swift.llm import deep_getattr
from swift.plugin import Tuner, extra_tuners, loss_mapping, optimizers_map
from swift.plugin.loss import cross_entropy_loss_func
from swift.tuners import LoraConfig, Swift
from swift.utils import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from swift.llm import TrainArguments


AUX_HEAD_CONFIG = 'aux_head_config.json'
AUX_TRAINABLES = 'aux_trainables.safetensors'
AUX_VIT_TRAINABLES = 'vit.safetensors'
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


def _is_vit_or_aligner_parameter_name(model_arch, name: str) -> bool:
    return _parameter_in_prefixes(name, model_arch.vision_tower + model_arch.aligner)


def _get_active_tasks(aux_config: Dict[str, object]) -> Sequence[str]:
    tasks_cfg = aux_config.get('tasks', {})
    return [task for task in AUX_TASKS if tasks_cfg.get(task, {}).get('enabled', True)]


def _is_zero_number(value: object) -> bool:
    if value is None:
        return False
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _get_loss_weight(task_weights: Mapping[str, object], task: str) -> float:
    weight = task_weights.get(task, 1.0)
    return 0.0 if _is_zero_number(weight) else float(weight)


def _get_task_train_weight(task_cfg: Mapping[str, object]) -> float:
    for key in ('train_weight', 'task_weight', 'loss_weight'):
        if key in task_cfg:
            weight = task_cfg[key]
            return 0.0 if _is_zero_number(weight) else float(weight)
    return 1.0


def _get_trainer_loss_scale(trainer, num_items_in_batch) -> float:
    if trainer is None or num_items_in_batch is None:
        return 1.0
    if (getattr(trainer.args, 'average_tokens_across_devices', False)
            and getattr(trainer, 'model_accepts_loss_kwargs', False)):
        return float(trainer.accelerator.num_processes)
    return 1.0


def _scalarize_loss_tensor(loss: torch.Tensor) -> torch.Tensor:
    if loss.ndim == 0:
        return loss
    if loss.numel() == 1:
        return loss.reshape(())
    return loss.mean()


def _extract_loss_tensor(loss_output: Any, task: str) -> Optional[torch.Tensor]:
    def _sum_tensors(value: Any) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return _scalarize_loss_tensor(value)
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
        return _scalarize_loss_tensor(loss_output)
    if isinstance(loss_output, Mapping):
        loss = loss_output.get('loss')
        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            return _scalarize_loss_tensor(loss)
        tensor = _sum_tensors(loss_output.get('weighted_loss_items'))
        if tensor is not None:
            return tensor
        for key in ('total_loss', 'final_loss', 'weighted_loss', 'origin_loss'):
            tensor = _sum_tensors(loss_output.get(key))
            if tensor is not None:
                return tensor
        if isinstance(loss, torch.Tensor):
            return _scalarize_loss_tensor(loss)
        raise TypeError(f'Loss mapping for task "{task}" must contain a tensor field named "loss".')
    loss = getattr(loss_output, 'loss', None)
    if isinstance(loss, torch.Tensor) and loss.numel() == 1:
        return _scalarize_loss_tensor(loss)
    tensor = _sum_tensors(get_value(loss_output, 'weighted_loss_items', None))
    if tensor is not None:
        return tensor
    for attr_name in ('total_loss', 'final_loss', 'weighted_loss', 'origin_loss'):
        tensor = _sum_tensors(getattr(loss_output, attr_name, None))
        if tensor is not None:
            return tensor
    if isinstance(loss, torch.Tensor):
        return _scalarize_loss_tensor(loss)
    raise TypeError(f'Unsupported auxiliary loss output type for task "{task}": {type(loss_output)!r}.')


def _sanitize_metric_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in name).strip('_')


def _to_metric_value(value: Any):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        return value.mean() if value.ndim > 0 else value
    if isinstance(value, Number):
        return value
    return None


def _iter_named_values(value: Any, prefix: str = ''):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key = _sanitize_metric_name(str(key))
            name = f'{prefix}_{key}' if prefix else key
            yield from _iter_named_values(nested, name)
        return
    if isinstance(value, (list, tuple)):
        for idx, nested in enumerate(value):
            name = f'{prefix}_{idx}' if prefix else str(idx)
            yield from _iter_named_values(nested, name)
        return
    if prefix:
        yield prefix, value


def _extract_loss_item_metrics(loss_output: Any) -> Dict[str, Any]:
    metrics = {}
    for field_name, suffix in (('loss_items', ''), ('weighted_loss_items', '_weighted')):
        loss_items = get_value(loss_output, field_name, None)
        if loss_items is None:
            continue
        for name, value in _iter_named_values(loss_items):
            metric_value = _to_metric_value(value)
            if metric_value is not None:
                metrics[f'{name}{suffix}'] = metric_value
    return metrics


def _attach_aux_state_to_outputs(outputs: Any, aux_state: Dict[str, Any]) -> None:
    try:
        setattr(outputs, AUX_STATE_ATTR, aux_state)
    except Exception:
        pass


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
    target_model = get_target_model(model)
    if getattr(target_model, '_qwen3vl_aux_patched', False):
        return model

    aux_config = aux_config or build_aux_config(target_model)
    _attach_task_modules(target_model, aux_config)
    reference_param = next(
        (param for name, param in target_model.named_parameters() if not _is_aux_parameter_name(name)), None)
    if reference_param is not None:
        target_model.aux_heads.to(device=reference_param.device, dtype=reference_param.dtype)
        target_model.aux_losses.to(device=reference_param.device, dtype=reference_param.dtype)
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
        if active_tasks:
            kwargs['output_hidden_states'] = True
        input_ids = kwargs.get('input_ids')
        if input_ids is None and args:
            input_ids = args[0]

        outputs = origin_forward(*args, **kwargs)
        aux_state = make_aux_state(active_tasks=active_tasks, plugin_config=self.aux_head_config)
        aux_predictions = {}
        set_value(outputs, AUX_STATE_ATTR, aux_state)
        set_value(outputs, 'aux_plugin_config', self.aux_head_config)
        set_value(outputs, 'aux_predictions', aux_predictions)
        set_value(outputs, 'aux_task_losses', aux_state['task_losses'])
        set_value(outputs, 'aux_loss', None)
        set_value(outputs, 'aux_enabled_tasks', list(active_tasks))
        set_value(outputs, 'aux_loss_weights', {})
        set_aux_state(self, aux_state)

        if not active_tasks:
            return outputs
        if input_ids is None:
            return outputs

        hidden_states = getattr(outputs, 'hidden_states', None)
        if not hidden_states:
            return outputs
        image_grid_thw = kwargs.get('image_grid_thw')
        if image_grid_thw is None:
            return outputs

        aux_features = {'hidden_states': hidden_states}
        task_weights = {
            task: _get_task_train_weight(self.aux_head_config['tasks'][task]) for task in active_tasks
        }
        aux_state['task_weights'] = task_weights
        set_value(outputs, 'aux_features', aux_features)
        set_value(outputs, 'visual_hidden_state_layers', list(self.aux_head_config['shared'].get('layer_indices', [])))
        set_value(outputs, 'aux_enabled_tasks', list(active_tasks))
        set_value(outputs, 'aux_loss_weights', task_weights)
        set_aux_state(self, aux_state)
        raw_targets = {'bev': y_bev, 'cog': y_cog, 'god': y_god}
        raw_targets.update(extra_task_targets)
        aux_labels = self.aux_label_loader(
            batch_size=input_ids.shape[0],
            batch_kwargs=extra_label_inputs,
            raw_targets=raw_targets,
            device=hidden_states[-1].device,
        )
        set_value(outputs, 'aux_labels', aux_labels)

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
        set_value(outputs, 'aux_head_outputs', head_outputs)

        for task in active_tasks:
            prediction = head_outputs.get(f'{task}_aux_head_output')
            if prediction is None:
                continue
            aux_predictions[task] = prediction
            set_value(outputs, f'{task}_prediction', prediction)

            try:
                loss_output = self.aux_losses(
                    task,
                    predictions=prediction,
                    targets=aux_labels['task_targets'][task],
                    aux_labels=aux_labels,
                    aux_features=aux_features,
                    model_outputs=outputs,
                    task_cfg=self.aux_head_config['tasks'][task],
                    shared_cfg=self.aux_head_config['shared'],
                )
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f'The auxiliary loss for task "{task}" is still an interface stub. '
                    f'Please implement {task.title()}AuxLoss.forward(...).') from exc
            loss_item_metrics = _extract_loss_item_metrics(loss_output)
            if loss_item_metrics:
                aux_state['loss_items'][task] = loss_item_metrics
                for name, metric_value in loss_item_metrics.items():
                    set_value(outputs, f'{task}_{name}', metric_value)
            task_loss = _extract_loss_tensor(loss_output, task)
            aux_state['task_losses'][task] = task_loss
            set_value(outputs, f'{task}_aux_loss', task_loss)
            set_aux_state(self, aux_state)
            _attach_aux_state_to_outputs(outputs, aux_state)

        loss_values = [loss for loss in aux_state['task_losses'].values() if loss is not None]
        if loss_values:
            total = loss_values[0]
            for extra in loss_values[1:]:
                total = total + extra
            aux_state['aux_total'] = total
            weighted_total = None
            for task, task_loss in aux_state['task_losses'].items():
                if task_loss is None:
                    continue
                loss_weight = _get_loss_weight(task_weights, task)
                if loss_weight == 0.0:
                    continue
                weighted_loss = loss_weight * task_loss
                weighted_total = weighted_loss if weighted_total is None else weighted_total + weighted_loss
            aux_state['weighted_aux_loss'] = weighted_total
            set_value(outputs, 'aux_loss', total)
            if weighted_total is not None:
                set_value(outputs, 'aux_weighted_loss', weighted_total)
                set_value(outputs, 'loss', weighted_total)
            set_aux_state(self, aux_state)
            _attach_aux_state_to_outputs(outputs, aux_state)
        return outputs

    target_model.forward = MethodType(forward, target_model)
    target_model._qwen3vl_aux_patched = True
    logger.info(f'Attached Qwen3-VL auxiliary plugin with config: {aux_config}')
    return model


def qwen3vl_aux_loss(outputs, labels, num_items_in_batch=None, trainer=None, **kwargs) -> torch.Tensor:
    target_model = get_target_model(trainer.model) if trainer is not None else None
    aux_state = get_aux_state(outputs, target_model, default_tasks=AUX_TASKS)
    aux_cfg = get_value(outputs, 'aux_plugin_config', None) or aux_state.get('plugin_config')
    lm_loss_weight = 1.0
    if aux_cfg is not None:
        lm_loss_weight = float(aux_cfg['shared'].get('lm_loss_weight', 1.0))
    task_losses = collect_task_losses_from_outputs(outputs, AUX_TASKS) or dict(aux_state.get('task_losses') or {})
    task_weights = get_value(outputs, 'aux_loss_weights', None) or dict(aux_state.get('task_weights') or {})
    weighted_aux_loss = get_value(outputs, 'aux_weighted_loss', None)
    if weighted_aux_loss is None:
        weighted_aux_loss = aux_state.get('weighted_aux_loss')
    loss_items = dict(aux_state.get('loss_items') or {})
    lm_loss = None
    if lm_loss_weight != 0:
        lm_loss = cross_entropy_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **kwargs)
        loss = lm_loss_weight * lm_loss
    else:
        anchor = weighted_aux_loss
        if anchor is None:
            anchor = get_value(outputs, 'logits', None)
        if anchor is None:
            anchor = next((task_loss for task_loss in task_losses.values() if isinstance(task_loss, torch.Tensor)),
                          None)
        if isinstance(anchor, torch.Tensor):
            loss = anchor.sum() * 0
            lm_loss = loss.detach()
        else:
            loss = torch.tensor(0.0)
            lm_loss = loss
    mode = None
    if trainer is not None:
        mode = 'train' if trainer.model.training else 'eval'
        trainer.custom_metrics[mode]['lm_loss'].update(lm_loss.detach())
        for task, item_metrics in loss_items.items():
            for name, metric_value in item_metrics.items():
                trainer.custom_metrics[mode][f'{task}_{name}'].update(metric_value)
    aux_total = None
    for task, task_loss in task_losses.items():
        if task_loss is None:
            continue
        if trainer is not None:
            trainer.custom_metrics[mode][f'{task}_aux_loss'].update(task_loss.detach())
        aux_total = task_loss if aux_total is None else aux_total + task_loss
        if weighted_aux_loss is None:
            loss_weight = _get_loss_weight(task_weights, task)
            if loss_weight == 0.0:
                continue
            task_loss = _scalarize_loss_tensor(task_loss.to(device=loss.device, dtype=loss.dtype))
            loss = loss + loss_weight * task_loss
    if isinstance(weighted_aux_loss, torch.Tensor):
        weighted_aux_loss = weighted_aux_loss.to(device=loss.device, dtype=loss.dtype)
        loss = loss + _scalarize_loss_tensor(weighted_aux_loss)
    if aux_total is None:
        aux_total = get_value(outputs, 'aux_loss', None)
    if aux_total is None:
        aux_total = aux_state.get('aux_total')
    loss = _scalarize_loss_tensor(loss)
    loss_scale = _get_trainer_loss_scale(trainer, num_items_in_batch)
    loss_for_trainer = loss / loss_scale
    if trainer is not None:
        if isinstance(aux_total, torch.Tensor):
            trainer.custom_metrics[mode]['aux_loss'].update(aux_total.detach())
        trainer.custom_metrics[mode]['objective_loss'].update(loss.detach())
        if loss_scale != 1.0:
            trainer.custom_metrics[mode]['loss_scale_factor'].update(loss_scale)
            trainer.custom_metrics[mode]['loss_returned_to_trainer'].update(loss_for_trainer.detach())
    return loss_for_trainer


class Qwen3VLAuxTuner(Tuner):

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        model_arch = model.model_meta.model_arch
        from swift.llm.train.tuner import get_modules_to_save, get_target_modules
        target_modules = get_target_modules(args, model)
        modules_to_save = get_modules_to_save(args, model, 'CAUSAL_LM')
        lora_kwargs = {
            'r': args.lora_rank,
            'target_modules': target_modules,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'bias': args.lora_bias,
            'modules_to_save': modules_to_save,
            'use_rslora': args.use_rslora,
            'use_dora': args.use_dora,
            'lorap_lr_ratio': args.lorap_lr_ratio,
            'init_lora_weights': args.init_weights,
        }
        if args.target_parameters is not None:
            lora_kwargs['target_parameters'] = args.target_parameters
        lora_config = LoraConfig(task_type='CAUSAL_LM', lora_dtype=args.lora_dtype, **lora_kwargs)
        logger.info(f'lora_config: {lora_config}')
        model = Swift.prepare_model(model, lora_config)
        model = attach_auxiliary_modules(model)

        train_vision = _is_true('QWEN3VL_AUX_TRAIN_VISION')
        train_vit = train_vision or args.vit_lr is not None or not args.freeze_vit
        train_aligner = (train_vision or _is_true('QWEN3VL_AUX_TRAIN_ALIGNER') or args.aligner_lr is not None
                         or not args.freeze_aligner)
        if train_vit:
            for module_prefix in model_arch.vision_tower:
                deep_getattr(model, module_prefix).requires_grad_(True)
            logger.info('Enabled full-parameter training for vision tower.')
        if train_aligner:
            for module_prefix in model_arch.aligner:
                deep_getattr(model, module_prefix).requires_grad_(True)
            logger.info('Enabled full-parameter training for aligner.')
        return model

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        trainable_parameter_names = {name for name, param in model.named_parameters() if param.requires_grad}
        if state_dict is None:
            state_dict = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        model.save_pretrained(save_directory, state_dict=state_dict, safe_serialization=safe_serialization, **kwargs)

        target_model = get_target_model(model)
        aux_config = getattr(target_model, 'aux_head_config', build_aux_config(target_model))
        with open(os.path.join(save_directory, AUX_HEAD_CONFIG), 'w', encoding='utf-8') as f:
            json.dump(aux_config, f, ensure_ascii=False, indent=2)

        aux_state_dict = {}
        for name, value in state_dict.items():
            if name in trainable_parameter_names and _is_aux_parameter_name(name):
                aux_state_dict[name] = value
        if aux_state_dict:
            safetensors.torch.save_file(
                aux_state_dict, os.path.join(save_directory, AUX_TRAINABLES), metadata={'format': 'pt'})

        model_arch = model.model_meta.model_arch
        vit_state_dict = {
            name: value
            for name, value in state_dict.items()
            if name in trainable_parameter_names and _is_vit_or_aligner_parameter_name(model_arch, name)
        }
        if vit_state_dict:
            safetensors.torch.save_file(
                vit_state_dict, os.path.join(save_directory, AUX_VIT_TRAINABLES), metadata={'format': 'pt'})

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        is_trainable = bool(kwargs.get('is_trainable', False))
        load_aux_heads = _is_true('QWEN3VL_AUX_LOAD_HEADS', 'true' if is_trainable else 'false')
        model = Swift.from_pretrained(model, model_id, **kwargs)

        vit_weights_path = os.path.join(model_id, AUX_VIT_TRAINABLES)
        if os.path.exists(vit_weights_path):
            state_dict = safetensors.torch.load_file(vit_weights_path)
            _, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(f'Loaded ViT/aligner trainables from {vit_weights_path}.')
            if unexpected:
                logger.warning(f'Unexpected keys when loading ViT/aligner trainables: {unexpected}')

        if not load_aux_heads:
            return model

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
