"""Module intelligence.py"""
import datasets
import transformers

import src.elements.variable as vr
import src.modelling.t5.metrics
import src.modelling.t5.model
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

        # Pre-trained Model: To graphics processing unit, if available
        self.__model = src.modelling.t5.model.Model(variable=variable, parameters=self.__parameters).exc()
        self.__model.to(self.__variable.DEVICE)

    def __data_collator(self) -> transformers.DataCollatorForSeq2Seq:
        """

        :return:
        """

        return transformers.DataCollatorForSeq2Seq(
            tokenizer=self.__parameters.tokenizer, model=self.__parameters.checkpoint)

    def __call__(self, data: datasets.DatasetDict) -> transformers.Seq2SeqTrainer:
        """
        Switch to ray.tune.Tuner

        :param data: The data; tokenized.
        :return:
        """

        trainable = transformers.Seq2SeqTrainer(
            model=self.__model,
            args=self.__settings.args(),
            train_dataset=data['train'],
            eval_dataset=data['validate'],
            tokenizer=self.__parameters.tokenizer,
            data_collator=self.__data_collator(),
            compute_metrics=self.__metrics.exc
        )

        trainable.train()

        return trainable
