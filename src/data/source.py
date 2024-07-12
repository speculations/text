import logging
import datasets


class Source:

    def __init__(self):
        """
        Constructor
        """

        # The Data
        self.__dataset: datasets.DatasetDict = datasets.load_dataset('billsum')

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __temporary(self):

        return self.__dataset['ca_test']

    def exc(self):

        # The data segments
        self.__logger.info('The data segments:\n%s', self.__dataset.keys())

        # The initial focus
        self.__logger.info('Focusing on data segment <ca_test>\n')
        temporary: datasets.Dataset = self.__temporary()
        self.__logger.info(temporary.__dir__())

        return temporary
