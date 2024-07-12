import logging

import datasets

import src.modelling.t5.splittings


class Steps:

    def __init__(self, splits: datasets.DatasetDict):
        """

        :param splits:
        """

        self.__splits = splits

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):
        """

        :return:
        """

        self.__splits: datasets.DatasetDict = src.modelling.t5.splittings.Splittings(splits=self.__splits).__call__()
        self.__logger.info(self.__splits.keys())
