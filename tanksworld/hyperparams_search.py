import my_config as cfg
from centralized_util import CentralizedTraining
from optuna.pruners import BasePruner
from env_config import update_config
import numpy as np
from datetime import datetime
import logging
import sys
import os
import optuna


def sample_ppo_params(trial: optuna.Trial):
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    net_arch_size = trial.suggest_categorical("net_arch_size", [32, 64, 128])
    extract_ftr_model = trial.suggest_categorical("extract_ftr_model", ["small", "big"])
    #    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    #    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128])

    return {
        "batch_size": batch_size,
        "net_arch_size": net_arch_size,
        "extract_ftr_model": extract_ftr_model,
    }


def objective(trial):
    params = cfg.params
    update_config(params)
    ppo_params = sample_ppo_params(trial)
    desc = ";".join([key + ":" + str(value) for key, value in ppo_params.items()])
    desc = datetime.now().strftime("%y-%m-%d-%H:%M:%S") + "-" + desc
    params.update(ppo_params)
    centralized_training = CentralizedTraining(trial=trial, exp_desc=desc, **params)
    centralized_training.train()
    return centralized_training.score


class CustomPruner(BasePruner):
    def __init__(self, warmup_steps=1000000, prune_threshold=0.1):
        self.warmup_steps = warmup_steps
        self.prune_threshold = prune_threshold

    def prune(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> bool:
        # Get the latest score reported from this trial
        step = trial.last_step

        if step:  # trial.last_step == None when no scores have been reported yet
            this_score = trial.intermediate_values[step]
            if step > self.warmup_steps and this_score < self.prune_threshold:
                print("Prune this trials, step {}, score {}".format(step, this_score))
                return True

        return False


if __name__ == "__main__":
    params = cfg.params
    if not os.path.exists(params["exp_dir"]):
        os.makedirs(params["exp_dir"])
    study_name = "trials"  # Unique identifier of the study.
    storage_name = "sqlite:///{}/trials.db".format(params["exp_dir"])
    search_space = {
        "batch_size": [32, 64],
        "net_arch_size": [64, 128],
        "extract_ftr_model": ["small", "big"],
    }
    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler(search_space),
        pruner=CustomPruner(params["warmup_steps"], params["prune_threshold"]),
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)
