import logging

import datasets
import transformers

import src.elements.variable as vr

import src.modelling.t5.splittings
import src.modelling.t5.intelligence


class Steps:

    def __init__(self, splits: datasets.DatasetDict):
        """

        :param splits:
        """

        self.__splits = splits

        # A set of values for machine learning model development
        self.__variable = vr.Variable()
        self.__variable._replace(EPOCHS=2)

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):
        """

        :return:
        """

        # T5 tokenized training & testing splits
        data: datasets.DatasetDict = src.modelling.t5.splittings.Splittings(splits=self.__splits).__call__()
        self.__logger.info(data.keys())

        # intelligence = src.modelling.t5.intelligence.Intelligence(variable=self.__variable)
        # model: transformers.Seq2SeqTrainer = intelligence(data=data)
        # self.__logger.info(model.__dir__())
        # model.save_model()
