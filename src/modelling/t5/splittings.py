"""Module splittings.py"""
import datasets

import config
import src.modelling.t5.preprocessing


class Splittings:
    """
    This class splits a datasets.Dataset into tokenized training & testing sets
    """

    def __init__(self, source: datasets.Dataset):
        """

        :param source: A datasets.Dataset data piece
        """

        self.__source = source

        # Configurations
        self.__configurations = config.Config()

        # Instances
        self.__preprocessing = src.modelling.t5.preprocessing.Preprocessing()

    def __preliminary(self) -> datasets.DatasetDict:
        """
        Splitting the data into training & testing sets; each split is a datasets.arrow_dataset.Dataset, and
        each instance of a split has keys -> 'text', 'summary', 'title'

        :return:
        """

        splits: datasets.DatasetDict = self.__source.train_test_split(
            test_size=self.__configurations.test_fraction)

        return splits

    def __call__(self) -> datasets.DatasetDict:
        """

        :return:
        """

        # Data splitting
        splits: datasets.DatasetDict = self.__preliminary()

        # Converting each split into a tokenized split
        return splits.map(self.__preprocessing.exc, batched=True)
