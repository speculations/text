"""Module interface.py"""
import logging

import datasets

import src.modelling.t5.steps


class Interface:
    """
    An interface to one or more model development packages
    """

    def __init__(self, source: datasets.DatasetDict, device: str):
        """

        :param source: A datasets.DatasetDict for modelling
        :param device:
        """

        self.__source = source
        self.__device = device


        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):
        """

        :return:
        """

        src.modelling.t5.steps.Steps(source=self.__source, device=self.__device).exc()
