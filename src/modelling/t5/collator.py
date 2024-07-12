
import transformers

import src.modelling.t5.parameters

class Collator:

    def __init__(self):

        self.__parameters = src.modelling.t5.parameters.Parameters()

    def __call__(self):

        return transformers.DataCollatorForSeq2Seq(
            tokenizer=self.__parameters.tokenizer, model=self.__parameters.checkpoint)
