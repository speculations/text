"""Module interface.py"""
import logging

import datasets

import config
import src.modelling.t5.steps



class Interface:
    """
    An interface to one or more model development packages
    """

    def __init__(self, source: datasets.Dataset):
        """

        :param source: A datasets.Dataset data piece for modelling
        """

        self.__source = source

        # Splitting
        self.__configurations = config.Config()
        self.__splits: datasets.DatasetDict = self.__source.train_test_split(
            test_size=self.__configurations.test_fraction)

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):
        """

        :return:
        """

        src.modelling.t5.steps.Steps(splits=self.__splits).exc()
