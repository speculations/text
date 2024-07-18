import logging

import datasets
import transformers

import src.elements.variable as vr
import src.modelling.t5.intelligence
import src.modelling.t5.preprocessing


class Steps:

    def __init__(self, source: datasets.DatasetDict, device: str):
        """

        :param source: A dictionary of data splits; training, validation, etc., splits.
        :param device:
        """

        self.__source = source
        self.__device = device

        # A set of values for machine learning model development
        self.__variable = vr.Variable()
        self.__variable._replace(EPOCHS=2)

        # Preprocessing Instance: For tokenization.
        self.__preprocessing = src.modelling.t5.preprocessing.Preprocessing(variable=self.__variable)

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):
        """
        model.save_model()

        :return:
        """

        # Converting each split into a T5 tokenized split
        data: datasets.DatasetDict = self.__source.map(self.__preprocessing.exc, batched=True)
        self.__logger.info(self.__source.keys())
        self.__logger.info(data.keys())

        # Model
        intelligence = src.modelling.t5.intelligence.Intelligence(
            variable=self.__variable, device=self.__device)
        model: transformers.Seq2SeqTrainer = intelligence(data=data)
        self.__logger.info(model.__dir__())
