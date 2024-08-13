import os
import ray
import ray.tune
import ray.tune.schedulers as rts
import transformers

import src.elements.variable as vr


class Settings:

    def __init__(self, variable: vr.Variable):
        """

        :param variable:
        """

        self.__variable = variable

        self.__perturbation_interval = 2

        self.hp_space = {
            'per_device_train_batch_size': 32,
            'per_device_eval_batch_size': 32,
            'num_train_epochs': ray.tune.choice([2, 3, 4, 5])
        }

    def scheduler(self):

        scheduler = rts.PopulationBasedTraining(
            time_attr='training_iteration',
            metric='',
            mode='max',
            perturbation_interval=self.__perturbation_interval,
            hyperparam_mutations={
                'learning_rate': ray.tune.uniform(lower=5e-3, upper=1e-1),
                'weight_decay': ray.tune.uniform(lower=0.0, upper=0.25),
                'per_device_train_batch_size': [16, 32, 64]
            },
            quantile_fraction=0.25,
            resample_probability=0.25
        )

        return scheduler

    def args(self) -> transformers.Seq2SeqTrainingArguments:
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
