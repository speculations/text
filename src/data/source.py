import logging
import datasets


class Source:

    def __init__(self):
        """
        Constructor
        """

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

        return self.__dataset['test']

    def exc(self):
        """

        :return:
        """

        # The data segments
        self.__logger.info('The data segments:\n%s', self.__dataset.keys())
        self.__logger.info('Training Set:\n%s', self.__dataset['train'].shape)
        self.__logger.info('Validate Set:\n%s', self.__dataset['validate'].shape)
        self.__logger.info('Test Set:\n%s', self.__dataset['test'].shape)

        # The initial focus
        temporary: datasets.Dataset = self.__temporary()
        self.__logger.info('Focusing on data segment <test>\n%s\n%s', type(temporary), temporary.features)

        return temporary
