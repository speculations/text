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

        temporary = self.__splits['test'].train_test_split(test_size=0.25)

        cuts = datasets.DatasetDict({
            'train': self.__splits['train'],
            'valid': temporary['train'],
            'test': temporary['test']
        })
        self.__logger.info(cuts.keys())
        self.__logger.info('Training:\n%s', cuts['train'].shape)
        self.__logger.info('Validating:\n%s', cuts['valid'].shape)
        self.__logger.info('Testing:\n%s', cuts['test'].shape)

        # src.modelling.t5.steps.Steps(splits=self.__splits).exc()
