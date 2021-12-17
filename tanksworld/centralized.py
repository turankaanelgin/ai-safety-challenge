from arena5.core.env_process import EnvironmentProcess
from collections import deque
from tanksworld.make_env import make_env
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union
import numpy as np
from stable_baselines3 import PPO
import stable_baselines3 as sb3
from matplotlib import pyplot as plt
import gym
import cv2
import my_config as cfg
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import torch.nn as nn
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from torchsummary import summary
from tanksworld.env_centralized.env import TanksWorldEnv
import torch
from torchsummary import summary
import sys
import os
import yaml
from datetime import datetime
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
)
from tanksworld.centralized_util import CustomCNN, TensorboardCallback, CustomMonitor
from tanksworld.env_centralized.minimap_util import displayable_rgb_map





if __name__ == '__main__':  
    args = cfg.args
    params = vars(args)
    #env_params = {"exe": args.exe, "training_tanks": [0],"static_tanks":[], "random_tanks":[5,6,7,8,9], "disable_shooting":[1,2,3,4,5,6,7,8,9]}
    if params['config'] == 1:
        params['env_params'] = {"exe": args.exe, "training_tanks": [0],"static_tanks":[1,2,3,4,5,6,7,8,9], "random_tanks":[], 
                'friendly_fire':True, 'take_damage_penalty':False, 'kill_bonus':True, 'death_penalty':False,
                "disable_shooting":[1,2,3,4,5,6,7,8,9], 'penalty_weight': params['penalty_weight']}
        params['config_desc'] = 'froze and disable shooting tank 1->9'
    elif params['config'] == 2:
        params['env_params'] = {"exe": args.exe, "training_tanks": [0],"static_tanks":[], "random_tanks":[1,2,3,4,5,6,7,8,9], 
                'friendly_fire':True, 'take_damage_penalty':False, 'kill_bonus':True, 'death_penalty':False,
                "disable_shooting":[], 'penalty_weight': params['penalty_weight']}
        params['config_desc'] = 'random tank 1->9'

    centralized_training = CentralizedTraining(**params)
    if args.record:
        centralized_training.record(args.video_path)
    elif args.training:
        centralized_training.train()
    elif args.eval_mode:
        centralized_training.eval()

    #else:#training
