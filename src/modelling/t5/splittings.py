"""Module splittings.py"""
import datasets

import src.modelling.t5.preprocessing


class Splittings:
    """
    This class splits a datasets.Dataset into T5 tokenized training & testing sets
    """

    def __init__(self, source: datasets.DatasetDict):
        """

        :param source: Training, ...
        """

        self.__source = source

        # Instances
        self.__preprocessing = src.modelling.t5.preprocessing.Preprocessing()

    def __call__(self) -> datasets.DatasetDict:
        """

        :return:
        """

        # Converting each split into a T5 tokenized split
        return self.__source.map(self.__preprocessing.exc, batched=True)
