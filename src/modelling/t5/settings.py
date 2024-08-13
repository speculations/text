import ray
import ray.tune
import ray.tune.schedulers as rts


class Settings:

    perturbation_interval = 2

    scheduler = rts.PopulationBasedTraining(
        time_attr='training_iteration',
        metric='',
        mode='max',
        perturbation_interval=perturbation_interval,
        hyperparam_mutations={
            'learning_rate': ray.tune.uniform(lower=5e-3, upper=1e-1),
            'weight_decay': ray.tune.uniform(lower=0.0, upper=0.25),
            'per_device_train_batch_size': [16, 32, 64]
        },
        quantile_fraction=0.25,
        resample_probability=0.25
    )

    hp_space = {
        'per_device_train_batch_size': 32,
        'per_device_eval_batch_size': 32,
        'num_train_epochs': ray.tune.choice([2, 3, 4, 5])
    }
