import logging

import transformers

import src.elements.variable as vr
import src.modelling.t5.parameters


class Model:

    def __init__(self, variable: vr.Variable):
        """

        :param variable:
        """

        self.__variable = variable
        self.__parameters = src.modelling.t5.parameters.Parameters()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self) -> transformers.models.t5.modeling_t5.T5ForConditionalGeneration:

        # Configurations
        config = transformers.GenerationConfig.from_pretrained(
            pretrained_model_name=self.__parameters.checkpoint, **{'max_new_tokens': self.__variable.MAX_NEW_TOKENS})
        self.__logger.info('max_length: %s', config.max_length)
        self.__logger.info('max_new_tokens: %s', config.max_new_tokens)

        # Model initialisation
        model: transformers.models.t5.modeling_t5.T5ForConditionalGeneration
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.checkpoint, config=config)

        return model
