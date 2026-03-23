# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import nullcontext
from copy import deepcopy
import inspect
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import gather_object
from transformers import PreTrainedModel
from trl import SFTTrainer as HFSFTTrainer

from swift.llm import disable_gradient_checkpointing
from swift.trainers.rlhf_trainer.gkd_trainer import DataSource, GKDTrainer
from swift.trainers.rlhf_trainer.rollout_mixin import DataType
from swift.trainers.rlhf_trainer.utils import patch_profiling_context, patch_profiling_decorator
from swift.utils import get_logger, unwrap_model_for_generation

logger = get_logger()


class OPSDTrainer(GKDTrainer):
    """External OPSD trainer kept separate from ms-swift source registration."""

    _REFERENCE_KEYS = ('reference', 'ground_truth', 'solution', 'label', 'answer')

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        args = kwargs.get('args')
        self.reference_placeholder = getattr(args, 'reference_placeholder', '<reference>') if args is not None else '<reference>'
        self.opsd_teacher_mode = getattr(args, 'opsd_teacher_mode', 'snapshot') if args is not None else 'snapshot'
        if args is not None:
            if getattr(args, 'lmbda', 1.0) != 1.0:
                logger.info('OPSD enforces lmbda=1.0, overriding args.lmbda=%s', args.lmbda)
            args.lmbda = 1.0
            if getattr(args, 'seq_kd', False):
                logger.warning('OPSD does not support seq_kd. Forcing seq_kd=False.')
                args.seq_kd = False
            if getattr(args, 'use_liger_kernel', False):
                raise NotImplementedError('OPSD currently does not support `use_liger_kernel`.')

        teacher_model = kwargs.get('teacher_model')
        self._use_student_as_teacher = teacher_model is None
        if teacher_model is None:
            if model is None:
                raise ValueError('OPSD requires `model` when `teacher_model` is not provided.')
            teacher_model = deepcopy(model)
            teacher_model.requires_grad_(False)
            kwargs['teacher_model'] = teacher_model
            if self.opsd_teacher_mode == 'shared':
                logger.info('OPSD teacher will share student weights during loss computation (stop-grad branch).')
            else:
                logger.info('OPSD teacher will use a frozen snapshot initialized from student weights.')

        super().__init__(model, *_args, **kwargs)
        if self._use_student_as_teacher and self.opsd_teacher_mode == 'shared':
            teacher_snapshot = self.teacher_model
            self.teacher_model = self.model
            if self.args.offload_teacher_model:
                logger.warning('offload_teacher_model is ignored when OPSD teacher shares student weights.')
                self.args.offload_teacher_model = False
            if teacher_snapshot is not self.model:
                del teacher_snapshot
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    @classmethod
    def _extract_reference(cls, sample: Dict[str, Any]) -> Optional[str]:
        for key in cls._REFERENCE_KEYS:
            if key in sample and sample[key] is not None:
                value = sample[key]
                if isinstance(value, str):
                    value = value.strip()
                    if value:
                        return value
                elif isinstance(value, (int, float)):
                    return str(value)
        return None

    def _replace_reference_placeholder(self, messages: List[Dict[str, Any]], reference: Optional[str],
                                       *, use_reference: bool) -> bool:
        found_placeholder = False
        replacement = reference or '' if use_reference else ''
        for message in messages:
            if message.get('role') != 'user':
                continue
            content = message.get('content')
            if not isinstance(content, str):
                continue
            if self.reference_placeholder not in content:
                continue
            found_placeholder = True
            message['content'] = content.replace(self.reference_placeholder, replacement)
        return found_placeholder

    @staticmethod
    def _extract_response_token_ids(labels: torch.Tensor, eos_token_id: Optional[int]) -> List[List[int]]:
        response_token_ids = []
        for row in labels:
            ids = row[row != -100].tolist()
            if not ids and eos_token_id is not None:
                ids = [eos_token_id]
            response_token_ids.append(ids)
        return response_token_ids

    @staticmethod
    def _ensure_last_assistant_message(messages: List[Dict[str, Any]]) -> None:
        if not messages or messages[-1].get('role') != 'assistant':
            messages.append({'role': 'assistant', 'content': None})

    def _accumulate_seen_tokens(self, model_inputs: Dict[str, Any]) -> None:
        attention_mask = model_inputs.get('attention_mask')
        if attention_mask is None:
            return
        local_tokens = attention_mask.sum().to(dtype=torch.long)
        if self.accelerator.num_processes > 1:
            gathered_tokens = self.accelerator.gather(local_tokens.reshape(1))
            total_tokens = gathered_tokens.sum().item()
        else:
            total_tokens = local_tokens.item()
        self._last_step_tokens = int(total_tokens)
        current_seen = getattr(self.state, 'num_input_tokens_seen', 0) or 0
        self.state.num_input_tokens_seen = current_seen + self._last_step_tokens

    def _update_rollout_metrics(self, response_token_ids: List[List[int]]) -> None:
        if not response_token_ids:
            return
        lengths = [len(ids) for ids in response_token_ids]
        if lengths:
            self.custom_metrics['train']['rollout_response_len'].update(sum(lengths) / len(lengths))
        repeat_ratios = []
        eos_token_id = getattr(self.processing_class, 'eos_token_id', None)
        eos_hits = 0
        valid_eos_count = 0
        for ids in response_token_ids:
            if len(ids) > 1:
                adjacent_repeats = sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1])
                repeat_ratios.append(adjacent_repeats / (len(ids) - 1))
            if eos_token_id is not None and ids:
                valid_eos_count += 1
                eos_hits += int(ids[-1] == eos_token_id)
        if repeat_ratios:
            self.custom_metrics['train']['rollout_repeat_ratio'].update(sum(repeat_ratios) / len(repeat_ratios))
        if valid_eos_count > 0:
            self.custom_metrics['train']['rollout_eos_rate'].update(eos_hits / valid_eos_count)

    def _filter_model_inputs(self, model: nn.Module, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        allowed_keys = {
            'input_ids',
            'attention_mask',
            'position_ids',
            'text_position_ids',
            'token_type_ids',
            'inputs_embeds',
            'pixel_values',
            'pixel_values_videos',
            'image_grid_thw',
            'video_grid_thw',
            'labels',
            'logits_to_keep',
            'loss_scale',
            'loss_scale_mask',
        }
        processing_class = getattr(self, 'processing_class', None)
        if processing_class is not None:
            allowed_keys.update(getattr(processing_class, 'model_input_names', []) or [])

        try:
            forward_params = inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            return {k: v for k, v in model_inputs.items() if k in allowed_keys}
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in forward_params.values()):
            return {k: v for k, v in model_inputs.items() if k in allowed_keys}
        allowed_keys.update(forward_params)
        return {k: v for k, v in model_inputs.items() if k in allowed_keys}

    def _build_student_prompt_inputs(self, source_inputs: DataType, references: List[Optional[str]]) -> DataType:
        student_inputs = deepcopy(source_inputs)
        for data, reference in zip(student_inputs, references):
            messages = deepcopy(data.get('messages', []))
            self._replace_reference_placeholder(messages, reference, use_reference=False)
            data['messages'] = messages
        return student_inputs

    def _build_teacher_rollout_inputs(self, source_inputs: DataType, generated_inputs: DataType,
                                      references: List[Optional[str]]) -> DataType:
        teacher_inputs = deepcopy(source_inputs)
        for data, generated_data, reference in zip(teacher_inputs, generated_inputs, references):
            messages = deepcopy(data.get('messages', []))
            found_placeholder = self._replace_reference_placeholder(messages, reference, use_reference=True)
            if reference and not found_placeholder:
                warning_once = getattr(logger, 'warning_once', logger.warning)
                warning_once(
                    f'OPSD: reference is provided but `{self.reference_placeholder}` placeholder is not found in user content.'
                )
            self._ensure_last_assistant_message(messages)
            data['messages'] = messages
            for key in ('response_token_ids', 'response_loss_mask', 'rollout_infos', 'rollout_logprobs', 'finish_reason',
                        'is_truncated', 'add_eos'):
                if key in generated_data:
                    data[key] = deepcopy(generated_data[key])
        return teacher_inputs

    @patch_profiling_decorator
    def training_step(self,
                      model: nn.Module,
                      inputs: DataType,
                      num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        args = self.args
        source_inputs = deepcopy(inputs)

        with patch_profiling_context(self, 'get_completions'):
            if self.template.truncation_strategy == 'raise':
                source_inputs = self.resample_encode_failed_inputs(source_inputs)
            references = [self._extract_reference(sample) for sample in source_inputs]
            student_source_inputs = self._build_student_prompt_inputs(source_inputs, references)

            if args.use_vllm:
                processed_inputs = self._preprocess_inputs(student_source_inputs)
                generated_inputs = self._fast_infer(processed_inputs)
                response_token_ids = [data.get('response_token_ids', []) for data in generated_inputs]
                self._update_rollout_metrics(response_token_ids)
                if self.log_completions:
                    messages = [inp['messages'][:-1] for inp in generated_inputs]
                    completions = [deepcopy(inp['messages'][-1]['content']) for inp in generated_inputs]
                    valid_messages = gather_object(messages)
                    valid_completions = gather_object(completions)
                    self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(valid_messages))
                    self._logs['completion'].extend(valid_completions)
            else:
                prompt_inputs = self._prepare_batch_inputs(student_source_inputs, encode_prompt_only=True)
                with unwrap_model_for_generation(
                        model, self.accelerator,
                        gather_deepspeed3_params=args.ds3_gather_for_generation) as unwrapped_model:
                    unwrapped_model.eval()
                    generation_inputs = dict(prompt_inputs)
                    generation_model_inputs = {k: v for k, v in generation_inputs.items() if k != 'labels'}
                    generation_model_inputs = self._filter_model_inputs(unwrapped_model, generation_model_inputs)
                    generation_inputs = dict(generation_model_inputs)
                    _, _, generated_labels = self.generate_on_policy_outputs(
                        unwrapped_model, generation_inputs, self.generation_config, self.processing_class.pad_token_id)
                    unwrapped_model.train()

                response_token_ids = self._extract_response_token_ids(
                    generated_labels, getattr(self.processing_class, 'eos_token_id', None))
                self._update_rollout_metrics(response_token_ids)
                generated_inputs = deepcopy(student_source_inputs)
                for data, token_ids in zip(generated_inputs, response_token_ids):
                    self._ensure_last_assistant_message(data['messages'])
                    data['response_token_ids'] = token_ids

            student_inputs = self._prepare_batch_inputs(generated_inputs, encode_prompt_only=False)
            teacher_rollout_inputs = self._build_teacher_rollout_inputs(source_inputs, generated_inputs, references)
            teacher_inputs = self._prepare_batch_inputs(teacher_rollout_inputs, encode_prompt_only=False)
            self._accumulate_seen_tokens(student_inputs)
            student_inputs['_data_source'] = DataSource.STUDENT
            student_inputs['_teacher_model_inputs'] = teacher_inputs

        with self.template.forward_context(self.model, student_inputs):
            return HFSFTTrainer.training_step(self, model, student_inputs, num_items_in_batch)

    @patch_profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        teacher_inputs = inputs.pop('_teacher_model_inputs', None)
        inputs.pop('_data_source', None)
        if teacher_inputs is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        student_model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}
        if self.args.sft_alpha > 0:
            student_model_inputs['labels'] = inputs['labels']
        student_model_inputs = self._filter_model_inputs(model, student_model_inputs)
        outputs_student = model(**student_model_inputs)

        teacher_model_inputs = {k: v for k, v in teacher_inputs.items() if k not in {'prompt', 'labels'}}
        if self._use_student_as_teacher:
            teacher_model = model
            load_context = nullcontext()
        else:
            teacher_model = self.teacher_model
            load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
        teacher_model_inputs = self._filter_model_inputs(teacher_model, teacher_model_inputs)

        was_training = teacher_model.training
        with torch.no_grad(), load_context, disable_gradient_checkpointing(teacher_model,
                                                                           self.args.gradient_checkpointing_kwargs):
            teacher_model.eval()
            outputs_teacher = teacher_model(**teacher_model_inputs)
        if was_training:
            teacher_model.train()

        shifted_student_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)
        shifted_teacher_labels = torch.roll(teacher_inputs['labels'], shifts=-1, dims=1)
        student_mask = shifted_student_labels != -100
        teacher_mask = shifted_teacher_labels != -100
        shifted_student_logits = outputs_student.logits[student_mask][None]
        shifted_teacher_logits = outputs_teacher.logits[teacher_mask][None]

        if shifted_student_logits.shape[1] != shifted_teacher_logits.shape[1]:
            min_tokens = min(shifted_student_logits.shape[1], shifted_teacher_logits.shape[1])
            logger.warning('Token count mismatch between student (%s) and teacher (%s), truncating to %s.',
                           shifted_student_logits.shape[1], shifted_teacher_logits.shape[1], min_tokens)
            shifted_student_logits = shifted_student_logits[:, :min_tokens]
            shifted_teacher_logits = shifted_teacher_logits[:, :min_tokens]

        stu_dim = shifted_student_logits.shape[-1]
        tea_dim = shifted_teacher_logits.shape[-1]
        if stu_dim < tea_dim:
            shifted_student_logits = F.pad(shifted_student_logits, (0, tea_dim - stu_dim), 'constant', 0)
            shifted_student_logits[..., stu_dim:] = shifted_teacher_logits[..., stu_dim:]
        elif stu_dim > tea_dim:
            shifted_teacher_logits = F.pad(shifted_teacher_logits, (0, stu_dim - tea_dim), 'constant', 0)
            shifted_teacher_logits[..., tea_dim:] = shifted_student_logits[..., tea_dim:]

        loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            beta=self.beta,
        )
        if self.args.sft_alpha > 0:
            loss = loss + self.args.sft_alpha * outputs_student.loss

        if return_outputs:
            return (loss, outputs_student)
        return loss
