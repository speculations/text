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

    def __init__(self, variable: vr.Variable, device: str, output_directory: str):
        """
        self.__model.generate(max_new_tokens=self.__variable.MAX_NEW_TOKENS)

        :param variable: A set of values for machine learning model development
        :param device: 'cuda' or 'cpu'
        :param output_directory:
        """

        self.__variable = variable
        self.__output_directory = output_directory

        # Instances
        self.__metrics = src.modelling.t5.metrics.Metrics()
        self.__parameters = src.modelling.t5.parameters.Parameters()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

        # Initialising model
        config = transformers.GenerationConfig.from_pretrained(pretrained_model_name=self.__parameters.checkpoint)
        config.max_new_tokens = self.__variable.MAX_NEW_TOKENS
        self.__model: transformers.models.t5.modeling_t5.T5ForConditionalGeneration = transformers.AutoModelForSeq2SeqLM.from_pretrained(
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
            output_dir=self.__output_directory,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=self.__variable.LEARNING_RATE,
            per_device_train_batch_size=self.__variable.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.__variable.VALIDATE_BATCH_SIZE,
            weight_decay=0.01,
            num_train_epochs=self.__variable.EPOCHS,
            save_total_limit=2,
            load_best_model_at_end=True,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False
        )

    def __call__(self, data: datasets.DatasetDict) -> transformers.Seq2SeqTrainer:
        """

        :param data:
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

        trainer.train()

        return trainer
