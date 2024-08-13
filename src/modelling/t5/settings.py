import ray
import ray.tune
import ray.tune.schedulers as rts


class Settings:

    perturbation_interval = 2

    tune_by = {
        'per_device_train_batch_size': 32,
        'per_device_eval_batch_size': 32,
        'num_train_epochs': ray.tune.choice([2, 3, 4, 5])
    }

    rts.PopulationBasedTraining(
        time_attr='training_iteration',
        perturbation_interval=perturbation_interval,

    )
