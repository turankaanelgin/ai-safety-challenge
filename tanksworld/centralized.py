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

class CentralizedTraining():
    def __init__(self, **params):
        self.params = params
        desc = datetime.now().strftime("%y-%m-%d-%H:%M:%S") +'nstep{}-nenv{}'.format(1,2)
        if args.debug:
            self.save_path = './testing/'+ desc
        else:
            self.save_path = './results/'+ desc

        self.training_env = self.create_env()
        #check_env(self.training_env)
        self.eval_env = self.training_env
        #check_env(self.env)
        self.model = self.create_model()

    def create_env(self):
        def create_env_():
            return TanksWorldEnv(**self.params['env_params'])
        #print(self.params)
        #return create_env_()
        env = make_vec_env(create_env_, n_envs=self.params['n_envs'], vec_env_cls=DummyVecEnv)
        env = CustomMonitor(env, self.params['n_envs'])
        return env

    def create_model(self):
        if args.save_path is not None:
            model = PPO.load(args.save_path)
        else:
            policy_kwargs = {}
            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=512),
                net_arch=[dict(pi=[512], vf=[512])]
            )
            def linear_schedule(initial_value: float) -> Callable[[float], float]:
                def func(progress_remaining: float) -> float:
                    return progress_remaining * initial_value

                return func
            model = PPO("CnnPolicy", self.training_env, policy_kwargs=policy_kwargs, n_steps=self.params['n_steps'], learning_rate=linear_schedule(0.0003), verbose=2, batch_size=32, tensorboard_log=self.save_path)
        return model
         

    def record(self, save_video_path):
        observation = self.eval_env.reset()
        step = 0
        old_step =0
        env_count = 0
        game = 0
        observation_list = []
        while game < args.n_episode:
            action, _ = self.model.predict(observation)
            #action[0]=1#forward
            #action[1]=0#left right 
            #action[2]=0#shoot
            observation, reward, done, info = self.eval_env.step(action)
            observation0 = observation.transpose(1,2,0)
            tanks_img = displayable_rgb_map(observation0)
            #img = env.overview_map()
            fig, axes = plt.subplots(1, 1)
            #axes[0].imshow(tanks_img)
            #axes[1].imshow(img)
            #axes[0].imshow(tanks_img)
            plt.imshow(tanks_img)
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
            plt.close()
            observation_list.append(data)
            if done:
                step = 0
                game += 1
                observation = self.eval_env.reset()
        out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*"MJPG"), 5, (640, 480), True)
        for img in observation_list:
            #img = cv2.cvtColor(img)
            out.write(img)
        out.release()

    def train(self): 
        os.mkdir(self.save_path)
        with open(self.save_path+'/config.yaml', 'w') as file:
            yaml.dump(args.__dict__,file)

        
        if self.params['save_path'] is not None:
            model = PPO.load(self.params['save_path'], env=env)
        else:
            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=512),
                net_arch=[dict(pi=[512], vf=[512])]
            )




        checkpoint_callback = CheckpointCallback(save_freq=self.params['save_freq'], save_path=self.save_path + '/checkpoints', name_prefix='rl_model')
        tensorboard_callback = TensorboardCallback()
        callback_list=[]
        #callback_list = [checkpoint_callback]
        callback_list = [checkpoint_callback, tensorboard_callback]
        self.model.learn(total_timesteps=self.params['timestep'], callback=callback_list)



if __name__ == '__main__':  
    args = cfg.args
    params = vars(args)
    #env_params = {"exe": args.exe, "training_tanks": [0],"static_tanks":[], "random_tanks":[5,6,7,8,9], "disable_shooting":[1,2,3,4,5,6,7,8,9]}
    params['env_params'] = {"exe": args.exe, "training_tanks": [0],"static_tanks":[1,2,3,4,5,6,7,8,9], "random_tanks":[], "disable_shooting":[1,2,3,4,5,6,7,8,9]}
    centralized_training = CentralizedTraining(**params)
    if args.record:
        centralized_training.record(args.video_path)
    elif args.training:
        centralized_training.train()
    elif args.eval_mode:
        from  os.path import join as pjoin
        import json
        model_path = pjoin(args.save_path, 'checkpoints', args.checkpoint)
        print(model_path)
        model = PPO.load(model_path)
        env = create_env()
        observation = env.reset()
        step = 0
        old_step =0
        env_count = 0
        game = 0
        episode_statistics = []
        mean_statistics = {}
        std_statistics = {}
        all_statistics = {}
        while True:
            action, _ = model.predict(observation)
            #observation, reward, done, info = env.step(np.random.rand(15))
            observation, reward, done, info = env.step(action)
            step += 1
            #done = True
            if done:
                #print('=============== Time step ',  step - old_step)
                #print(info['average'])
                info['average']['step'] = step - old_step
                episode_statistics.append(info['average'])
                old_step = step
                observation = env.reset()
                game += 1
                if game == args.eval_game:
                    break
        for key in episode_statistics[0]:
            list_of_stats = list(episode_statistics[idx][key] for idx in range(len(episode_statistics)))
            mean_statistics[key] = np.average(list_of_stats)
            std_statistics[key] = np.std(list_of_stats)
            all_statistics[key] = list_of_stats

        with open(pjoin(args.save_path, 'mean_statistics.json'), 'w+') as f:
            json.dump(mean_statistics, f, indent=True)
        with open(pjoin(args.save_path, 'std_statistics.json'), 'w+') as f:
            json.dump(std_statistics, f, indent=True)

    #else:#training
