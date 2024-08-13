"""Module steps.py"""
import logging
import os

import datasets
import transformers

import config
import src.elements.variable as vr
import src.modelling.t5.parameters as pr
import src.modelling.t5.depositories
import src.modelling.t5.intelligence
import src.modelling.t5.preprocessing


class Steps:
    """
    Class Steps
    """

    def __init__(self, source: datasets.DatasetDict, device: str):
        """

        :param source: A dictionary of data splits; training, validation, etc., splits.
        :param device: A string denoting graphics or central processing unit, i.e., 'cuda' or 'cpu', respectively.
        """

        self.__source = source

        # A set of values for machine learning model development
        self.__variable = vr.Variable()
        self.__variable = self.__variable._replace(
            EPOCHS=2,
            MODEL_OUTPUT_DIRECTORY=os.path.join(config.Config().warehouse, 't5'),
            DEVICE=device)

        # Parameters
        self.parameters = pr.Parameters()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __after_tokenization(self) -> datasets.DatasetDict:

        # Preprocessing Instance: For tokenization.
        preprocessing = src.modelling.t5.preprocessing.Preprocessing(variable=self.__variable, parameters=self.parameters)

        # Converting each split into a T5 tokenized split
        data: datasets.DatasetDict = self.__source.map(preprocessing.exc, batched=True)
        self.__logger.info('source: %s\ndata: %s', self.__source.keys(), data.keys())

        return data

    def exc(self):
        """
        model.save_model()

        :return:
        """

        # Re-write
        src.modelling.t5.depositories.Depositories().exc(path=self.__variable.MODEL_OUTPUT_DIRECTORY)

        # The data, after tokenization
        data = self.__after_tokenization()

        # Model
        intelligence = src.modelling.t5.intelligence.Intelligence(variable=self.__variable, parameters=self.parameters)
        model: transformers.Seq2SeqTrainer = intelligence(data=data)
        self.__logger.info(dir(model))
