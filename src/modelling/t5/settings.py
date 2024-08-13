import ray
import ray.tune
import ray.tune.schedulers as rts


class Settings:

    perturbation_interval = 2
    lr = 0.1

    tune_by = {
        'per_device_train_batch_size': 32,
        'per_device_eval_batch_size': 32,
        'num_train_epochs': ray.tune.choice([2, 3, 4, 5])
    }

    rts.PopulationBasedTraining(
        time_attr='training_iteration',
        metric='',
        mode='max',
        perturbation_interval=perturbation_interval,
        hyperparam_mutations={
            'lr': ray.tune.qloguniform(lower=5e-3, upper=1e-1, q=5e-4),
            'h0': ray.tune.uniform(lower=0.0, upper=1.0),
            'h1': ray.tune.uniform(lower=0.0, upper=1.0)
        },
        quantile_fraction=0.25,
        resample_probability=0.25

    )


