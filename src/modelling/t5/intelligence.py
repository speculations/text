
import transformers

import src.elements.variable as vr
import src.modelling.t5.parameters

class Intelligence:

    def __init__(self, variable: vr.Variable):
        """

        """

        self.__variable = variable

        self.__parameters = src.modelling.t5.parameters.Parameters()

        self.__model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.checkpoint)

        self.__args = transformers.Seq2SeqTrainingArguments(
            output_dir='bills',
            eval_strategy='epoch',
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

    def __call__(self):

        transformers.Seq2SeqTrainer(
            model=self.__model,
            args=self.__args

        )

