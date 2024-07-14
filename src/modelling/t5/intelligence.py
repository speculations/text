"""Module intelligence.py"""
import logging
import datasets
import transformers

import src.elements.variable as vr
import src.modelling.t5.metrics
import src.modelling.t5.parameters


class Intelligence:
    """
    The model development class.
    """

    def __init__(self, variable: vr.Variable):
        """

        :param variable: A set of values for machine learning model development
        """

        self.__variable = variable

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

        # Instances
        self.__metrics = src.modelling.t5.metrics.Metrics()
        self.__parameters = src.modelling.t5.parameters.Parameters()

        # Initialising model, and initiating GPU (graphics processing unit) based model
        # development, if possible.  Test {'max_length': ...}
        self.__model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.checkpoint, **{'max_length': 128})

        self.__model.to(self.__parameters.device)

        # Collator
        self.__data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer=self.__parameters.tokenizer, model=self.__parameters.checkpoint)

        # Arguments
        self.__args = transformers.Seq2SeqTrainingArguments(
            output_dir='bills',
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=self.__variable.LEARNING_RATE,
            per_device_train_batch_size=self.__variable.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.__variable.TEST_BATCH_SIZE,
            weight_decay=0.01,
            num_train_epochs=self.__variable.EPOCHS,
            save_total_limit=3,
            load_best_model_at_end=True,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False
        )

    def __call__(self, data: datasets.DatasetDict) -> transformers.Seq2SeqTrainer:

        trainer = transformers.Seq2SeqTrainer(
            model=self.__model,
            args=self.__args,
            train_dataset=data['train'],
            eval_dataset=data['test'],
            tokenizer=self.__parameters.tokenizer,
            data_collator=self.__data_collator,
            compute_metrics=self.__metrics.exc
        )

        trainer.train()

        return trainer
