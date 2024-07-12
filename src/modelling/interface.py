import logging
import datasets

import config

class Interface:

    def __init__(self, source: datasets.Dataset):
        """

        :param source:
        """

        self.__source = source

        # Configurations
        self.__configurations = config.Config()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __split(self):
        """

        :return:
        """

        splits: datasets.DatasetDict = self.__source.train_test_split(
            test_size=self.__configurations.test_fraction)

        self.__logger.info(splits.keys())
        self.__logger.info(splits['train'].shape)
        self.__logger.info(splits['train'][0].keys())

        return splits

    def exc(self):
        """

        :return:
        """

        splits = self.__split()
        self.__logger.info('Text:\n%s', splits['train'][0]['text'])
        self.__logger.info('Title:\n%s', splits['train'][0]['title'])
        self.__logger.info('Summary:\n%s', splits['train'][0]['summary'])
