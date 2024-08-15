"""Module intelligence.py"""
import logging
import os

import datasets
import transformers

import src.elements.variable as vr
import src.modelling.t5.metrics
import src.modelling.t5.skeleton
import src.modelling.t5.parameters as pr
import src.modelling.t5.settings


class Intelligence:
    """
    The model development class.
    """

    def __init__(self, variable: vr.Variable, parameters: pr.Parameters):
        """

        :param variable: A suite of values for machine learning
                         model development
        :param parameters: T5 specific parameters
        """

        self.__variable = variable
        self.__parameters = parameters

        # Setting: scheduler, arguments, ...
        self.__settings = src.modelling.t5.settings.Settings(variable=variable)

        # Instances
        self.__metrics = src.modelling.t5.metrics.Metrics(parameters=self.__parameters)

        # Configuration
        self.__skeleton = src.modelling.t5.skeleton.Skeleton(variable=variable, parameters=self.__parameters).exc()

    def __data_collator(self) -> transformers.DataCollatorForSeq2Seq:
        """

        :return:
        """

        return transformers.DataCollatorForSeq2Seq(
            tokenizer=self.__parameters.tokenizer, model=self.__parameters.checkpoint)

    def __model(self):
        """

        :return:
        """

        return transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.checkpoint, config=self.__skeleton
        )

    def __call__(self, data: datasets.DatasetDict):
        """
        https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer

        :param data: The data; tokenized.
        :return:
        """

        trainer = transformers.Seq2SeqTrainer(
            model_init=self.__model,
            args=self.__settings.args(),
            train_dataset=data['train'],
            eval_dataset=data['validate'],
            tokenizer=self.__parameters.tokenizer,
            data_collator=self.__data_collator(),
            compute_metrics=self.__metrics.exc
        )

        latest = trainer.hyperparameter_search(
            hp_space=lambda _: self.__settings.hp_space(),
            n_trials=self.__parameters.n_trials,
            resources_per_trial={'cpu': self.__parameters.n_cpu, 'gpu': self.__parameters.n_gpu},
            backend='ray',
            scheduler=self.__settings.scheduler(),
            keep_checkpoints_num=1,
            checkpoint_score_attr='training_iteration',
            progress_reporter=self.__settings.reporting(),
            storage_path=os.path.join(self.__variable.MODEL_OUTPUT_DIRECTORY, 'numerics'),
            name='robust',
            log_to_file=True
        )

        logging.info(type(latest))

        return latest
