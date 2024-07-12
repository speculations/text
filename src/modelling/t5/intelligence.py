
import transformers

import src.modelling.t5.parameters

class Intelligence:

    def __init__(self):
        """

        """

        self.__parameters = src.modelling.t5.parameters.Parameters()

        self.__model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.checkpoint)

        