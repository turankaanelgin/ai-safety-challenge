import my_config as cfg
from centralized_util import CentralizedTraining
from optuna.pruners import BasePruner
from optuna import pruners, samplers
from env_config import update_env_config
import numpy as np
from datetime import datetime
import logging
import sys
import os
import optuna
from optuna.samplers import TPESampler, CmaEsSampler


def sample_params_1tank(trial: optuna.Trial):
    return {
        #        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "batch_size": trial.suggest_categorical("batch_size", [64]),
        "net_arch_size": trial.suggest_categorical(
            "net_arch_size", [64, 128, 256, 512]
        ),
        "features_dim": trial.suggest_categorical("features_dim", [64, 128, 256]),
        "extract_ftr_model": trial.suggest_categorical(
            "extract_ftr_model", ["small", "medium"]
        ),
        #        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1),
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-4]),
        #        "penalty_weight": trial.suggest_float("penalty_weight", 0, 1),
        "penalty_weight": trial.suggest_categorical("penalty_weight", [0.4]),
        #        "n_steps": trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128]),
        "n_steps": trial.suggest_categorical("n_steps", [64]),
        "shot_reward": trial.suggest_categorical("shot_reward", [True, False]),
        "shot_reward_amount": trial.suggest_float(
            "shot_reward_amount", 1.0 / 500, 0.01
        ),
        #        "ent_coef": trial.suggest_loguniform("ent_coef", 0.00000001, 0.1),
        "ent_coef": trial.suggest_categorical("ent_coef", [0.0, 0.01]),
        "clip_range": trial.suggest_float(
            #            "clip_range", [0.1, 0.2, 0.4, 0.6, 0.8]
            #            "clip_range", [0.2, 0.4, 0.8]
            "clip_range",
            0.05,
            0.4,
        ),
    }


def sample_params_2tanks(trial: optuna.Trial):
    return {
        "features_dim": trial.suggest_categorical("features_dim", [128, 256]),
        "shot_reward": trial.suggest_categorical("shot_reward", [True, False]),
        "shot_reward_amount": trial.suggest_float("shot_reward_amount", 0.0001, 0.1),
        "clip_range": trial.suggest_float("clip_range", 0.001, 0.4),
    }


def sample_params_5tanks(trial: optuna.Trial):
    return {
        "shot_reward_amount": trial.suggest_float("shot_reward_amount", 0.00001, 0.05),
        "clip_range": trial.suggest_float("clip_range", 0.0001, 0.4),
    }
def sample_params_5vs5tanks(trial: optuna.Trial):
    return {
        "net_arch_size": trial.suggest_categorical("features_dim", [512, 1024]),
        "shot_reward_amount": trial.suggest_float("shot_reward_amount", 0.00001, 0.02),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4),
        "penalty_weight": trial.suggest_float("penalty_weight", 0.2, 0.8),
        "n_steps": trial.suggest_categorical("n_steps", [32, 64, 128]),
        "clip_range": trial.suggest_float("clip_range", 0.0001, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.01),
    }



def get_experiment_config(experiment):
    if experiment == "single-tank":
        return {"function": sample_params_1tank, "env_config": 2}
    elif experiment == "2vs1":
        return {"function": sample_params_2tanks, "env_config": 3}
    elif experiment == "5vs1":
        return {"function": sample_params_5tanks, "env_config": 8}
    elif experiment == "5vs5":
        return {"function": sample_params_5vs5tanks, "env_config": 9}
    else:
        raise Exception


def objective(trial):
    params = cfg.params
    exp_config = get_experiment_config(params["experiment"])
    params["config"] = exp_config["env_config"]
    ppo_params = exp_config["function"](trial)
    desc = ";".join([key + ":" + str(value) for key, value in ppo_params.items()])
    desc = datetime.now().strftime("%y-%m-%d-%H:%M:%S") + "-" + desc
    params.update(ppo_params)
    update_env_config(params)
    try:
        centralized_training = CentralizedTraining(trial=trial, exp_desc=desc, **params)
        centralized_training.train()
    except:
        import traceback

        traceback.print_exc()
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
    if not os.path.exists(params["experiment"]):
        os.makedirs(params["experiment"])
    study_name = "trials"  # Unique identifier of the study.
    storage_name = "sqlite:///{}/trials.db".format(params["experiment"])
    search_type = "grid_search"
    grid_search = False
    if grid_search:
        search_space = {
            "batch_size": [32, 64],
            "net_arch_size": [64, 128],
            "extract_ftr_model": ["small", "big"],
        }
        sampler = optuna.samplers.GridSampler(search_space)
    else:
        #        sampler = samplers.RandomSampler()
        sampler = TPESampler()
    #        sampler = samplers.CmaEsSampler()
    study = optuna.create_study(
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=params['warmup_steps']),
        #        pruner=CustomPruner(params["warmup_steps"], params["prune_threshold"]),
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)
