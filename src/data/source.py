import logging
import datasets


class Source:

    def __init__(self):

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self):

        dataset: datasets.DatasetDict = datasets.load_dataset('billsum')
        self.__logger.info(type(dataset))
        self.__logger.info(dataset.keys())

