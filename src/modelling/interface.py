import logging
import datasets

import config

class Interface:

    def __init__(self):

        self.__configurations = config.Config()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, source: datasets.Dataset):

        splits: datasets.DatasetDict = source.train_test_split(test_size=self.__configurations.test_fraction)
        self.__logger.info(splits.keys())
