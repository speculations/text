"""Module splittings.py"""
import datasets

import src.modelling.t5.preprocessing


class Splittings:
    """
    This class splits a datasets.Dataset into T5 tokenized training & testing sets
    """

    def __init__(self, splits: datasets.DatasetDict):
        """

        :param splits: Training, ...
        """

        self.__splits = splits

        # Instances
        self.__preprocessing = src.modelling.t5.preprocessing.Preprocessing()

    def __call__(self) -> datasets.DatasetDict:
        """

        :return:
        """

        # Converting each split into a T5 tokenized split
        return self.__splits.map(self.__preprocessing.exc, batched=True)
