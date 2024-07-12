"""Module interface.py"""
import logging

import datasets

import src.modelling.t5.splittings


class Interface:
    """
    An interface to one or more model development packages
    """

    def __init__(self, source: datasets.Dataset):
        """

        :param source: A datasets.Dataset data piece
        """

        self.__source = source
        self.__splits: datasets.DatasetDict = src.modelling.t5.splittings.Splittings(source=source).__call__()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):
        """

        :return:
        """

        self.__logger.info(self.__splits.keys())
