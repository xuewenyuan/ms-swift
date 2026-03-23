import os
from dataclasses import dataclass
from typing import List, Literal, Optional

from swift.llm.argument.rlhf_args import GRPOArguments, RLHFArguments, rlhf_support_vllm_types
from swift.llm.argument.train_args import TrainArguments
from swift.utils import get_current_device, get_logger, is_master, is_mp, json_parse_to_dict

logger = get_logger()


@dataclass
class OPSDArguments(RLHFArguments):
    rlhf_type: Literal['opsd'] = 'opsd'
    reference_placeholder: str = '<reference>'
    opsd_teacher_mode: Literal['snapshot', 'shared'] = 'snapshot'

    def __post_init__(self):
        self._process_loss_type()
        self._init_rm()
        self._init_simpo()
        self._init_max_completion_length()
        self._init_opsd_padding_side()
        self._set_opsd_default()
        self._init_opsd_rollout()
        self._init_teacher_deepspeed()

        original_rlhf_type = self.rlhf_type
        self.rlhf_type = 'gkd'
        try:
            GRPOArguments.__post_init__(self)
            TrainArguments.__post_init__(self)
        finally:
            self.rlhf_type = original_rlhf_type

        self._check_sequence_parallel()
        self._check_padding_free()
        self._check_opsd()

        if self.loss_scale is None:
            self.loss_scale = 'last_round'
        if isinstance(self.ref_adapters, str):
            self.ref_adapters = [self.ref_adapters]
        if self.ref_model is not None:
            raise ValueError('OPSD does not require a ref_model to be passed in.')

    def _init_opsd_padding_side(self):
        self.padding_side = 'left'

    def _set_opsd_default(self):
        self.remove_unused_columns = False
        logger.info(f'Setting args.remove_unused_columns: {self.remove_unused_columns}')
        if self.beta is None:
            self.beta = 0.5
        self.lmbda = 1.0
        self.seq_kd = False
        if self.reference_placeholder is None:
            self.reference_placeholder = '<reference>'

    def _init_opsd_rollout(self):
        if self.rlhf_type not in [*rlhf_support_vllm_types, 'opsd']:
            return

        if self.vllm_mode is not None and not self.use_vllm:
            raise ValueError('vllm_mode is not supported when use_vllm is false')
        if self.vllm_mode is None and self.use_vllm:
            raise ValueError('vllm_mode is required when use_vllm is true')
        self._init_external_vllm()

        if self.vllm_mode == 'server':
            assert not self.use_vllm or self.vllm_server_host is not None or self.vllm_server_base_url is not None

        if self.async_generate:
            raise NotImplementedError('Currently, async_generate is not supported for OPSD.')

        if not self.use_vllm and self.vllm_tensor_parallel_size != 1:
            self.vllm_tensor_parallel_size = 1
            logger.warning('set vllm_tensor_parallel_size to 1 since use_vllm false')
        self._external_vllm_warning()

    def _init_external_vllm(self):
        if self.rlhf_type != 'opsd' or (self.vllm_server_host is None and self.vllm_server_base_url is None):
            return
        from swift.trainers.rlhf_trainer.vllm_client import VLLMClient
        if is_master():
            logger.info('Start connecting to vLLM server')
            self.vllm_client = VLLMClient(
                base_urls=self.vllm_server_base_url,
                hosts=self.vllm_server_host,
                server_ports=self.vllm_server_port,
                group_ports=self.vllm_server_group_port,
                connection_timeout=self.vllm_server_timeout)
            self.vllm_client.close_communicator()
            self.vllm_client.init_communicator(device=get_current_device())
            logger.info('Connected to vLLM server')

    def _external_vllm_warning(self):
        if self.rlhf_type != 'opsd' or not self.vllm_server_host:
            return
        if self.vllm_max_model_len is not None:
            logger.warning(
                "Configuration conflict: 'vllm_max_model_len=%s' is ignored for external vLLM. "
                'Please specify it when launching the inference service: '
                '`swift rollout --vllm_max_model_len <value>`', self.vllm_max_model_len)

    def _check_padding_free(self):
        TrainArguments._check_padding_free(self)
        if self.padding_free or self.packing:
            supported_types = ['grpo', 'dpo', 'kto', 'gkd', 'opsd']
            if self.rlhf_type not in supported_types:
                raise NotImplementedError(
                    f"The current rlhf_type '{self.rlhf_type}' does not support padding_free/packing. "
                    'Please set --padding_free/packing to false.')

    def _check_sequence_parallel(self):
        if self.sequence_parallel_size > 1:
            raise NotImplementedError(
                f"The current rlhf_type '{self.rlhf_type}' does not support sequence_parallel. "
                'Please set --sequence_parallel_size to 1.')

    def _init_teacher_deepspeed(self):
        if not self.teacher_deepspeed:
            return
        ds_config_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'swift', 'llm', 'ds_config'))
        deepspeed_mapping = {
            name: f'{name}.json'
            for name in ['zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload']
        }
        for ds_name, ds_config in deepspeed_mapping.items():
            if self.teacher_deepspeed == ds_name:
                self.teacher_deepspeed = os.path.join(ds_config_folder, ds_config)
                break
        self.teacher_deepspeed = json_parse_to_dict(self.teacher_deepspeed)
        logger.info(f'Using teacher_deepspeed config: {self.teacher_deepspeed}')

    def _check_opsd(self):
        if self.teacher_model is None and self.teacher_deepspeed:
            logger.warning('teacher_deepspeed is ignored for OPSD when teacher_model is not set.')
        if is_mp() and self.use_vllm:
            raise ValueError('OPSD with vLLM is not compatible with `device_map`. '
                             'Please set NPROC_PER_NODE equal to num_processes.')
        if self.multi_turn_scheduler is not None:
            raise NotImplementedError('Currently, multi_turn_scheduler is not supported for OPSD.')
