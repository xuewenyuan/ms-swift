from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn


AUX_STATE_ATTR = '_qwen3vl_aux_state'
_LAST_AUX_STATE: Dict[str, Any] = {}
_MISSING = object()


def iter_wrapped_modules(model: nn.Module) -> Iterable[nn.Module]:
    queue = [model]
    visited = set()
    while queue:
        current = queue.pop(0)
        if not isinstance(current, nn.Module) or id(current) in visited:
            continue
        visited.add(id(current))
        yield current
        for attr_name in ('module', 'base_model', 'model'):
            try:
                wrapped = getattr(current, attr_name, None)
            except Exception:
                continue
            if isinstance(wrapped, nn.Module):
                queue.append(wrapped)


def get_target_model(model: nn.Module) -> nn.Module:
    modules = list(iter_wrapped_modules(model))
    for module in modules:
        if getattr(module, '_qwen3vl_aux_patched', False):
            return module
    for module in modules:
        if _owns_attr(module, 'lm_head') and hasattr(module, 'config'):
            return module
    return model


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    value = getattr(obj, key, _MISSING)
    if value is not _MISSING:
        return value
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return default


def set_value(obj: Any, key: str, value: Any) -> None:
    try:
        setattr(obj, key, value)
    except Exception:
        pass
    if key != 'loss':
        return
    try:
        obj[key] = value
    except Exception:
        pass


def collect_task_losses_from_outputs(outputs: Any, default_tasks: Sequence[str]) -> Dict[str, torch.Tensor]:
    task_losses = dict(get_value(outputs, 'aux_task_losses', {}) or {})
    active_tasks = get_value(outputs, 'aux_enabled_tasks', None) or default_tasks
    for task in active_tasks:
        if task in task_losses and task_losses[task] is not None:
            continue
        attr_loss = get_value(outputs, f'{task}_aux_loss', None)
        if isinstance(attr_loss, torch.Tensor):
            task_losses[task] = attr_loss
    return {task: loss for task, loss in task_losses.items() if isinstance(loss, torch.Tensor)}


def make_aux_state(*,
                   task_losses: Optional[Dict[str, torch.Tensor]] = None,
                   aux_total: Optional[torch.Tensor] = None,
                   weighted_aux_loss: Optional[torch.Tensor] = None,
                   loss_items: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
                   task_weights: Optional[Dict[str, float]] = None,
                   active_tasks: Optional[Sequence[str]] = None,
                   plugin_config: Optional[Dict[str, object]] = None) -> Dict[str, Any]:
    return {
        'task_losses': dict(task_losses or {}),
        'aux_total': aux_total,
        'weighted_aux_loss': weighted_aux_loss,
        'loss_items': {task: dict(items) for task, items in (loss_items or {}).items()},
        'task_weights': dict(task_weights or {}),
        'active_tasks': list(active_tasks or []),
        'plugin_config': plugin_config,
    }


def set_aux_state(model: nn.Module, state: Dict[str, Any]) -> None:
    setattr(model, AUX_STATE_ATTR, state)
    model._qwen3vl_last_aux_task_losses = state['task_losses']
    model._qwen3vl_last_aux_total_loss = state['aux_total']
    model._qwen3vl_last_aux_loss_weights = state['task_weights']
    model._qwen3vl_last_aux_enabled_tasks = state['active_tasks']
    _LAST_AUX_STATE.clear()
    _LAST_AUX_STATE.update(state)


def get_aux_state(outputs: Any = None,
                  model: Optional[nn.Module] = None,
                  default_tasks: Sequence[str] = ()) -> Dict[str, Any]:
    states = [_state_from_obj(outputs, default_tasks)]
    if model is not None:
        states.extend(_state_from_obj(module, default_tasks) for module in iter_wrapped_modules(model))
    if _LAST_AUX_STATE:
        states.append(
            make_aux_state(
                task_losses=_LAST_AUX_STATE.get('task_losses'),
                aux_total=_LAST_AUX_STATE.get('aux_total'),
                weighted_aux_loss=_LAST_AUX_STATE.get('weighted_aux_loss'),
                loss_items=_LAST_AUX_STATE.get('loss_items'),
                task_weights=_LAST_AUX_STATE.get('task_weights'),
                active_tasks=_LAST_AUX_STATE.get('active_tasks'),
                plugin_config=_LAST_AUX_STATE.get('plugin_config'),
            ))
    return _merge_aux_states(states)


def _owns_attr(module: nn.Module, attr_name: str) -> bool:
    return attr_name in module.__dict__ or attr_name in getattr(module, '_modules', {})


def _state_from_obj(obj: Any, default_tasks: Sequence[str]) -> Dict[str, Any]:
    state = get_value(obj, AUX_STATE_ATTR, None)
    if isinstance(state, Mapping):
        return make_aux_state(
            task_losses=state.get('task_losses'),
            aux_total=state.get('aux_total'),
            weighted_aux_loss=state.get('weighted_aux_loss'),
            loss_items=state.get('loss_items'),
            task_weights=state.get('task_weights'),
            active_tasks=state.get('active_tasks'),
            plugin_config=state.get('plugin_config'),
        )
    task_losses = collect_task_losses_from_outputs(obj, default_tasks)
    task_weights = get_value(obj, 'aux_loss_weights', None)
    aux_total = get_value(obj, 'aux_loss', None)
    active_tasks = get_value(obj, 'aux_enabled_tasks', None)
    plugin_config = get_value(obj, 'aux_plugin_config', None)
    if task_losses or task_weights or aux_total is not None or active_tasks or plugin_config is not None:
        return make_aux_state(
            task_losses=task_losses,
            aux_total=aux_total,
            weighted_aux_loss=get_value(obj, 'loss', None),
            task_weights=task_weights,
            active_tasks=active_tasks,
            plugin_config=plugin_config,
        )
    return {}


def _merge_aux_states(states: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    merged = make_aux_state()
    for state in states:
        if not state:
            continue
        if state.get('task_losses'):
            merged['task_losses'].update(state['task_losses'])
        if state.get('task_weights'):
            merged['task_weights'].update(state['task_weights'])
        if state.get('aux_total') is not None:
            merged['aux_total'] = state['aux_total']
        if state.get('weighted_aux_loss') is not None:
            merged['weighted_aux_loss'] = state['weighted_aux_loss']
        if state.get('loss_items'):
            for task, items in state['loss_items'].items():
                merged['loss_items'].setdefault(task, {}).update(items)
        if state.get('active_tasks'):
            merged['active_tasks'] = list(state['active_tasks'])
        if state.get('plugin_config') is not None:
            merged['plugin_config'] = state['plugin_config']
    return merged
