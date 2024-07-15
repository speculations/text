import logging
import datasets

import config

class Source:

    def __init__(self):
        """
        Constructor
        """

        self.__configurations = config.Config()

        # The Data
        self.__dataset: datasets.DatasetDict = datasets.load_dataset('billsum')
        validate = self.__dataset.pop('test')
        test = self.__dataset.pop('ca_test')
        self.__dataset['validate'] = validate
        self.__dataset['test'] = test

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __temporary(self):

        splits: datasets.DatasetDict = self.__dataset['test'].train_test_split(
            test_size=self.__configurations.fraction_validate)

        nodes = splits['test'].train_test_split(test_size=0.25)

        splittings = datasets.DatasetDict({
            'train': splits['train'],
            'validate': nodes['train'],
            'test': nodes['test']
        })

        return splittings

    def exc(self) -> datasets.DatasetDict:
        """

        :return:
        """

        # The data segments
        self.__logger.info('The data segments:\n%s', self.__dataset.keys())
        self.__logger.info('Training Set:\n%s', self.__dataset['train'].shape)
        self.__logger.info('Validate Set:\n%s', self.__dataset['validate'].shape)
        self.__logger.info('Test Set:\n%s', self.__dataset['test'].shape)

        # The initial focus
        temporary: datasets.DatasetDict = self.__temporary()
        self.__logger.info('Initially focusing on a small data segment\n%s\n%s', type(temporary), temporary.keys())
        self.__logger.info('Training:\n%s', temporary['train'].shape)
        self.__logger.info('Validating:\n%s', temporary['validate'].shape)
        self.__logger.info('Testing:\n%s', temporary['test'].shape)

        return temporary
