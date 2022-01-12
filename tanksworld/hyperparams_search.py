import my_config as cfg
from centralized_util import CentralizedTraining
import numpy as np
from env_config import update_config
import logging
import sys
import optuna

def sample_ppo_params(trial: optuna.Trial):
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
def objective(trial):
    params=sample_ppo_params(trial)
    return params['x']**2 + params['y'] **2


if __name__ == "__main__":
    args = cfg.args
    params = vars(args)
    update_config(params)
    centralized_training = CentralizedTraining(**params)

    study_name = "example-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    search_space = {"x": [0, 50], "y": [0, 99], 'batch_size': [32], 'n_steps':[8], 'learning_rate': [-1]}
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), 
            study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=20)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

