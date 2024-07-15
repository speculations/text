"""Module interface.py"""
import logging

import datasets

import src.modelling.t5.steps


class Interface:
    """
    An interface to one or more model development packages
    """

    def __init__(self, source: datasets.DatasetDict):
        """

        :param source: A datasets.DatasetDict for modelling
        """

        self.__source = source


        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):
        """

        :return:
        """

        src.modelling.t5.steps.Steps(splits=self.__source).exc()
