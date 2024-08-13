"""Module preprocessing.py"""
import datasets.formatting.formatting
import transformers

import src.elements.variable as vr
import src.modelling.t5.parameters as pr

class Preprocessing:
    """
    Class Preprocessing

    This class preprocesses a data batch, e.g., splits for model
    development, in line with T5 (Text-To-Text Transfer Transformer)
    architecture expectations.
    """

    def __init__(self, variable: vr.Variable, parameters: pr.Parameters):
        """

        :param variable: A suite of values for machine learning
                         model development
        """

        self.__variable = variable

        # The T5 specific parameters, and the T5 specific tokenizer
        self.__parameters = parameters
        self.__tokenizer: transformers.PreTrainedTokenizerFast = self.__parameters.tokenizer

    def exc(self, blob: datasets.formatting.formatting.LazyBatch) -> transformers.BatchEncoding:
        """

        :param blob: training or testing data batch
        :return:
        """

        # Independent Variable
        inputs = [self.__parameters.input_prefix + segment for segment in blob['text']]
        structure: transformers.BatchEncoding = self.__tokenizer(
            text=inputs, max_length=self.__variable.MAX_LENGTH_INPUT, truncation=True)

        # Dependent Variable; targets has a dictionary structure, wherein the keys are <input_ids> & <attention_mask>
        targets: transformers.BatchEncoding = self.__tokenizer(
            text_target=blob['summary'], max_length=self.__variable.MAX_LENGTH_TARGET, truncation=True)
        structure['labels']  = targets['input_ids']

        return structure
