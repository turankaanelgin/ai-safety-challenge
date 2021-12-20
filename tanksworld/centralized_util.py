from collections import deque
from typing import Callable
import numpy as np
from stable_baselines3 import PPO
import gym
import cv2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import torch.nn as nn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import torch
import os
import yaml
from datetime import datetime
from collections import deque
from typing import Callable
import numpy as np
from stable_baselines3 import PPO
from matplotlib import pyplot as plt
import gym
import cv2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import torch.nn as nn
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from tanksworld.env_centralized.env import TanksWorldEnv
import torch
import os
import yaml
from datetime import datetime
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
)
#from tanksworld.centralized_util import CustomCNN, TensorboardCallback, CustomMonitor
from tanksworld.env_centralized.minimap_util import displayable_rgb_map

class CentralizedTraining():
    def __init__(self, **params):
        self.params = params
        desc = datetime.now().strftime("%y-%m-%d-%H:%M:%S") \
                + 'TW-timestep{}M-nstep{}-nenv{}-timeout-{}-neg-{}-lrtype-{}-{}'.format(params['timestep']/1e6, params['n_steps'], 
                        params['n_envs'], params['env_params']['timeout'], params['penalty_weight'], params['lr_type'], params['config_desc'])
        if params['debug']:
            self.save_path = './testing/'+ desc
        else:
            self.save_path = './results/'+ desc

        self.training_env = self.create_env(self.params['n_envs'])
        #check_env(self.training_env)
        self.eval_env = self.create_env(1)
        #check_env(self.env)
        self.model = self.create_model()

    def create_env(self, n_envs):
        def create_env_():
            return TanksWorldEnv(**self.params['env_params'])
            #return gym.make('CarRacing-v0')
        if n_envs == 1:
            return create_env_()
        #print(self.params)
        #return create_env_()
        if self.params['dummy_proc']:
            env = make_vec_env(create_env_, n_envs=n_envs, vec_env_cls=DummyVecEnv)
        else:
            env = make_vec_env(create_env_, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

        env = VecFrameStack(env, 4)
        env = CustomMonitor(env, n_envs)
        return env

    def create_model(self):
        if self.params['save_path'] is not None and self.params['load_type'] == 'full':
            print('load model {}'.format(self.params['save_path']))
            model = PPO.load(self.params['save_path'], env=self.training_env)
        else:
            policy_kwargs = {}
            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=512),
                net_arch=[dict(pi=[64], vf=[64])]
            )
            def linear_schedule(initial_value: float) -> Callable[[float], float]:
                def func(progress_remaining: float) -> float:
                    return progress_remaining * initial_value

                return func

            if self.params['lr_type']=='linear':
                lr = linear_schedule(self.params['lr'])
            elif self.params['lr_type']=='constant':
                lr = self.params['lr']

            model = PPO("CnnPolicy", self.training_env, policy_kwargs=policy_kwargs, n_steps=self.params['n_steps'], 
                    learning_rate=lr, verbose=0, batch_size=64, ent_coef=self.params['ent_coef'], n_epochs=self.params['epochs'],
                    tensorboard_log=self.save_path)
            if self.params['save_path'] is not None and  self.params['load_type'] == 'cnn':
                loaded_model = PPO.load(self.params['save_path'])
                model.policy.features_extractor.load_state_dict(loaded_model.policy.features_extractor.state_dict())
            if self.params['freeze_cnn']:
                for param in model.policy.features_extractor.parameters():
                    param.requires_grad = False
        return model
         

    def record(self, save_video_path):
        observation = self.eval_env.reset()
        game = 0
        observation_list = []
        while game < self.params['n_episode']:
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
                game += 1
                observation = self.eval_env.reset()
        out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*"MJPG"), 5, (640, 480), True)
        for img in observation_list:
            #img = cv2.cvtColor(img)
            out.write(img)
        out.release()

    def train(self): 
        os.mkdir(self.save_path)
        with open(self.save_path+'/config.yaml', 'w') as f:
            yaml.dump(self.params, f)


        checkpoint_callback = CheckpointCallback(save_freq=self.params['save_freq'], save_path=self.save_path + '/checkpoints', name_prefix='rl_model')
        tensorboard_callback = TensorboardCallback()
        callback_list=[]
        #callback_list = [checkpoint_callback]
        callback_list = [checkpoint_callback, tensorboard_callback]
        self.model.learn(total_timesteps=self.params['timestep'], callback=callback_list)

    def eval(self):
        from  os.path import join as pjoin
        import json
        model_path = pjoin(args.save_path, 'checkpoints', args.checkpoint)
        print(model_path)
        model = PPO.load(model_path)
        env = create_env()
        observation = env.reset()
        step = 0
        old_step =0
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
                    s[key].append(stats[key]['value'])
            for key in s.keys():
                self.logger.record('{}/{}'.format(self.training_env.stats[0][key]['group'], key), np.mean(s[key]))

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
                    'dmg_inflict_on_enemy': {'value': infos[i]['red_stats']['damage_inflicted_on']['enemy'], 'group': '1_damage'},
                    'dmg_inflict_on_neutral': {'value':infos[i]['red_stats']['damage_inflicted_on']['neutral'],'group': '1_damage'}, 
                    'dmg_inflict_on_ally': {'value':infos[i]['red_stats']['damage_inflicted_on']['ally'],'group': '1_damage'},
                    'dmg_taken_by_ally': {'value':infos[i]['red_stats']['damage_taken_by']['ally'],'group': '1_damage'},
                    'dmg_taken_by_enemy': {'value':infos[i]['red_stats']['damage_taken_by']['enemy'],'group': '1_damage'},
                    '#shots':{'value':infos[i]['red_stats']["number_shots_fired"]["ally"],'group': '1_damage'},
                    'alive_ally':{'value':infos[i]['red_stats']["tanks_alive"]["ally"],'group': '2_lives'},
                    'alive_enemy':{'value':infos[i]['red_stats']["tanks_alive"]["enemy"],'group': '2_lives'},
                    'alive_neutral':{'value':infos[i]['red_stats']["tanks_alive"]["neutral"],'group': '2_lives'},
                    'step_per_episode':{'value':infos[i]['episode_step'],'group': '0_general_stats'},
                    'reward':{'value':self.rewards[i], 'group': '0_general_stats'},
                    })
                self.rewards[i] = 0
             
        return obs, rewards, dones, infos

    def close(self) -> None:
        return self.venv.close()