from collections import deque
from tanksworld.make_env import make_env
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union
import numpy as np
from stable_baselines3 import PPO
import stable_baselines3 as sb3
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
import torch
from torchsummary import summary
import sys
import os
import yaml
from datetime import datetime

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            #nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            #nn.ReLU(),
            #nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            #nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        return True

    #def _on_training_end(self) -> None:
    def _on_rollout_end(self) -> None:
        s = {}
        print('roll out end')
        if len(self.training_env.stats) > 0:
            for key in self.training_env.stats[0].keys():
                s[key] = []
            for stats in self.training_env.stats:
                for key in s.keys():
                    s[key].append(stats[key])
            for key in s.keys():
                self.logger.record('tanksworld stats/{}'.format(key), np.mean(s[key]))

class CustomMonitor(VecEnvWrapper):
#class TensorboardCallback():
    def __init__( self, venv: VecEnv, n_env):
        #super(CustomMonitor, self).__init__(venv)
        VecEnvWrapper.__init__(self, venv)
        self.stats = deque(maxlen=10)
        self.rewards = np.zeros(n_env)

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs


    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.rewards += rewards
        for i, done in enumerate(dones):
            if done:
                self.stats.append({
                    'dmg_inflict_on_enemy': infos[i]['red_stats']['damage_inflicted_on']['enemy'],
                    'dmg_inflict_on_neutral': infos[i]['red_stats']['damage_inflicted_on']['neutral'],
                    'dmg_inflict_on_ally': infos[i]['red_stats']['damage_inflicted_on']['ally'],
                    'dmg_taken_by_ally': infos[i]['red_stats']['damage_taken_by']['ally'],
                    'dmg_taken_by_enemy': infos[i]['red_stats']['damage_taken_by']['enemy'],
                    #'penalty_weight': infos[i]['reward_parameters']['penalty_weight'],
                    'reward':self.rewards[i]
                    })
                self.rewards[i] = 0
             
        return obs, rewards, dones, infos

    def close(self) -> None:
        return self.venv.close()
