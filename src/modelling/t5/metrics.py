import logging
import numpy as np
import evaluate

import src.modelling.t5.parameters

class Metrics:

    def __init__(self):

        self.__parameters = src.modelling.t5.parameters.Parameters()

        self.__tokenizer = self.__parameters.tokenizer

        self.__rouge = evaluate.load('rouge')

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, bucket):

        self.__logger.info('BUCKET:\n%s', type(bucket))
        predictions: np.ndarray
        labels: np.ndarray
        predictions, labels = bucket

        # Predictions: Skipping special tokens
        _predictions = self.__tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Labels, Active Labels: Re-visit.  Replacing ...
        active = np.where(labels != -100, labels, self.__tokenizer.pad_token_id)
        _labels = self.__tokenizer.batch_decode(active, skip_special_tokens=True)

        # Calculations: Initially, Aggregator = True
        calculations: dict = self.__rouge.compute(
            predictions=_predictions, references=_labels, use_aggregator=True, use_stemmer=True)
        self.__logger.info(calculations.keys())

        # Additionally
        lengths = [np.count_nonzero(prediction != self.__tokenizer.pad_token_id) for prediction in predictions]
        calculations['average'] = np.mean(lengths)
        self.__logger.info(calculations.keys())

        # Rounding
        return {key: round(value, 5) for key, value in calculations.items()}
