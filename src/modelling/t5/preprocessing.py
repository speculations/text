import logging

import datasets.formatting.formatting
import transformers

import src.elements.variable as vr
import src.modelling.t5.parameters as pr


class Preprocessing:

    def __init__(self, variable: vr.Variable):
        """

        :param variable:
        """

        self.__variable = variable

        # The T5 specific parameters, and the T5 specific tokenizer
        self.__parameters = pr.Parameters()
        self.__tokenizer: transformers.PreTrainedTokenizerFast = self.__parameters.tokenizer

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

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
