"""Module intelligence.py"""
import logging
import os.path

import datasets
import transformers

import src.elements.variable as vr
import src.modelling.t5.metrics
import src.modelling.t5.parameters


class Intelligence:
    """
    The model development class.
    """

    def __init__(self, variable: vr.Variable, device: str):
        """

        :param variable: A set of values for machine learning model development
        :param device: 'cuda' or 'cpu'
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

        # Configurations
        config = transformers.GenerationConfig.from_pretrained(
            pretrained_model_name=self.__parameters.checkpoint, **{'max_new_tokens': self.__variable.MAX_NEW_TOKENS})
        self.__logger.info('max_length: %s', config.max_length)
        self.__logger.info('max_new_tokens: %s', config.max_new_tokens)

        # Model initialisation
        self.__model: transformers.models.t5.modeling_t5.T5ForConditionalGeneration
        self.__model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.checkpoint, config=config)

        # To graphics processing unit, if available
        self.__model.to(device)

    def __data_collator(self) -> transformers.DataCollatorForSeq2Seq:
        """

        :return:
        """

        return transformers.DataCollatorForSeq2Seq(
            tokenizer=self.__parameters.tokenizer, model=self.__parameters.checkpoint)

    def __args(self) -> transformers.Seq2SeqTrainingArguments:
        """

        :return:
        """

        # Arguments
        return transformers.Seq2SeqTrainingArguments(
            output_dir=self.__variable.MODEL_OUTPUT_DIRECTORY,
            do_train=True,
            do_eval=True,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=self.__variable.LEARNING_RATE,
            weight_decay=0.01,
            per_device_train_batch_size=self.__variable.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.__variable.VALIDATE_BATCH_SIZE,
            num_train_epochs=self.__variable.EPOCHS,
            max_steps=-1,
            warmup_steps=0,
            logging_dir=os.path.join(self.__variable.MODEL_OUTPUT_DIRECTORY, '.logs'),
            save_total_limit=2,
            skip_memory_metrics=True,
            load_best_model_at_end=True,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False
        )

    def __call__(self, data: datasets.DatasetDict) -> transformers.Seq2SeqTrainer:
        """

        :param data: The data; tokenized.
        :return:
        """

        trainer = transformers.Seq2SeqTrainer(
            model=self.__model,
            args=self.__args(),
            train_dataset=data['train'],
            eval_dataset=data['validate'],
            tokenizer=self.__parameters.tokenizer,
            data_collator=self.__data_collator(),
            compute_metrics=self.__metrics.exc
        )

        # trainer.hyperparameter_search()

        trainer.train()

        return trainer
