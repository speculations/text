"""Module intelligence.py"""
import logging
import os.path

import datasets
import transformers

import src.elements.variable as vr
import src.modelling.t5.metrics
import src.modelling.t5.parameters

import src.modelling.t5.settings
import src.modelling.t5.model


class Intelligence:
    """
    The model development class.
    """

    def __init__(self, variable: vr.Variable, device: str):
        """

        :param variable: A set of values for machine learning model development
        :param device: 'cuda' or 'cpu'
        """

        self.__variable = variable

        # Setting: scheduler, arguments, ...
        self.__settings = src.modelling.t5.settings.Settings(variable=variable)


        # Instances
        self.__metrics = src.modelling.t5.metrics.Metrics()
        self.__parameters = src.modelling.t5.parameters.Parameters()


        # To graphics processing unit, if available
        self.__model = src.modelling.t5.model.Model(variable=variable).exc()
        self.__model.to(device)

    def __data_collator(self) -> transformers.DataCollatorForSeq2Seq:
        """

        :return:
        """

        return transformers.DataCollatorForSeq2Seq(
            tokenizer=self.__parameters.tokenizer, model=self.__parameters.checkpoint)



    def __call__(self, data: datasets.DatasetDict) -> transformers.Seq2SeqTrainer:
        """
        trainer.hyperparameter_search()

        :param data: The data; tokenized.
        :return:
        """

        trainer = transformers.Seq2SeqTrainer(
            model=self.__model,
            args=self.__settings.args(),
            train_dataset=data['train'],
            eval_dataset=data['validate'],
            tokenizer=self.__parameters.tokenizer,
            data_collator=self.__data_collator(),
            compute_metrics=self.__metrics.exc
        )

        trainer.train()

        return trainer
