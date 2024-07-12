import logging
import datasets

import config
import src.modelling.t5.preprocessing

class Interface:

    def __init__(self, source: datasets.Dataset):
        """

        :param source:
        """

        self.__source = source

        # Configurations
        self.__configurations = config.Config()

        # Instances
        self.__preprocessing = src.modelling.t5.preprocessing.Preprocessing()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __split(self) -> datasets.DatasetDict:
        """

        :return:
        """

        splits: datasets.DatasetDict = self.__source.train_test_split(
            test_size=self.__configurations.test_fraction)

        return splits

    def exc(self):
        """

        :return:
        """

        splits: datasets.DatasetDict = self.__split()
        self.__logger.info(splits.keys())
        self.__logger.info(splits['train'][0].keys())

        cuts: datasets.DatasetDict = splits.map(self.__preprocessing.exc, batched=True)
        self.__logger.info(cuts.keys())
