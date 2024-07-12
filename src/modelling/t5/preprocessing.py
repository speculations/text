import logging

import datasets.formatting.formatting
import transformers

import src.modelling.t5.parameters

class Preprocessing:

    def __init__(self):

        parameters = src.modelling.t5.parameters.Parameters()
        self.__tokenizer: transformers.PreTrainedTokenizerFast = parameters.tokenizer
        self.__max_length_input = parameters.max_length_input
        self.__max_length_target = parameters.max_length_target

        self.__prefix = 'summarize: '

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
        inputs = [self.__prefix + segment for segment in blob['text']]
        structure: transformers.BatchEncoding = self.__tokenizer(
            text=inputs, max_length=self.__max_length_input, truncation=True)

        # Dependent Variable; targets has a dictionary structure, wherein the keys are <input_ids> & <attention_mask>
        targets: transformers.BatchEncoding = self.__tokenizer(
            text_target=blob['summary'], max_length=self.__max_length_target, truncation=True)
        structure['labels']  = targets['input_ids']

        return structure
