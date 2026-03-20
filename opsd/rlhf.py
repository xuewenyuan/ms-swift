from typing import List, Optional, Union

from swift.ray import RayHelper
from swift.llm.infer import get_cached_dataset
from swift.llm.dataset.loader import DatasetLoader
from swift.utils import get_logger, get_model_parameter_info
from swift.llm.train.rlhf import SwiftRLHF

from .arguments import OPSDArguments
from .trainer import OPSDTrainer

logger = get_logger()


class SwiftOPSD(SwiftRLHF):
    args_class = OPSDArguments
    args: args_class

    @staticmethod
    def _set_main_input_name(model) -> None:
        if model is None:
            return
        model.main_input_name = 'input_ids'
        inner_model = getattr(model, 'model', None)
        if inner_model is not None:
            inner_model.main_input_name = 'input_ids'

    @RayHelper.function(group='default')
    def _prepare_dataset(self):
        args = self.args
        # OPSD needs raw `messages` during rollout, so encoding must be deferred to training.
        pre_process = False
        if args.cached_dataset or args.cached_val_dataset:
            assert not args.streaming, 'Cached dataset does not support streaming.'
            train_datasets, val_datasets = get_cached_dataset(self.args)
        else:
            train_datasets, val_datasets = [], []
        if args.dataset or args.val_dataset:
            train_dataset, val_dataset = self._get_dataset()
            train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset, pre_process=pre_process)
            if train_dataset is not None:
                train_datasets.append(train_dataset)
            if val_dataset is not None:
                val_datasets.append(val_dataset)
        train_dataset = DatasetLoader._concat_datasets(train_datasets)
        val_dataset = DatasetLoader._concat_datasets(val_datasets)
        if args.truncation_strategy != 'split':
            logger.info(f'train_dataset: {train_dataset}')
            logger.info(f'val_dataset: {val_dataset}')
        return [train_dataset, val_dataset]

    def _prepare_model_tokenizer(self):
        args = self.args
        self.ref_model = None
        self.value_model = None
        self.reward_model = None
        self.teacher_model = None

        if args.teacher_model is not None:
            result = self._prepare_single_model('teacher', 'teacher', args.teacher_model_type, args.teacher_model_revision)
            if result is not None:
                model, _ = result
                self.teacher_model = model
                self._set_main_input_name(self.teacher_model)

        from swift.llm.train.sft import SwiftSft
        SwiftSft._prepare_model_tokenizer(self)
        self._set_main_input_name(self.model)

    def _prepare_template(self) -> None:
        from swift.llm.train.sft import SwiftSft
        SwiftSft._prepare_template(self)
        self.template.set_mode('train')

    def _get_trainer_kwargs(self):
        trainer_kwargs = {}
        if self.teacher_model is not None:
            trainer_kwargs['teacher_model'] = self.teacher_model
        if self.args.use_vllm:
            trainer_kwargs['vllm_client'] = self.args.vllm_client
        if self.args.teacher_deepspeed:
            trainer_kwargs['teacher_deepspeed_config'] = self.args.teacher_deepspeed
        return trainer_kwargs

    @RayHelper.function(group='default')
    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()

        if args.task_type == 'seq_cls':
            args.problem_type = args.problem_type or getattr(self.model.config, 'problem_type', None)
            logger.info(f'args.problem_type: {args.problem_type}')
        args.save_args()

        data_collator = self._get_data_collator()
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        trainer = OPSDTrainer(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)


def opsd_main(args: Optional[Union[List[str], OPSDArguments]] = None):
    return SwiftOPSD(args).main()
