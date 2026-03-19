from typing import List, Optional, Union

from swift.ray import RayHelper
from swift.utils import get_logger, get_model_parameter_info
from swift.llm.train.rlhf import SwiftRLHF

from .arguments import OPSDArguments
from .trainer import OPSDTrainer

logger = get_logger()


class SwiftOPSD(SwiftRLHF):
    args_class = OPSDArguments
    args: args_class

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

        from swift.llm.train.sft import SwiftSft
        SwiftSft._prepare_model_tokenizer(self)

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
