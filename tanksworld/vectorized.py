from arena5.core.env_process import EnvironmentProcess
from collections import deque
from tanksworld.make_env import make_env
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
from env_original.make_env import make_env as make_env_origin
from env_stacked.make_env import make_env as make_env_stacked
from env_rgb.make_env import make_env as make_env_rgb
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
    def __init__( self, venv: VecEnv):
        super(CustomMonitor, self).__init__(venv)
        self.stats = deque(maxlen=10)

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs


    def step_wait(self) -> VecEnvStepReturn:
        #self.logger.record('sometest', 1.2)
        obs, rewards, dones, infos = self.venv.step_wait()
        for i, done in enumerate(dones):
            if done:
                self.stats.append({
                    'dmg_inflict_on_enemy': infos[i]['red_stats']['damage_inflicted_on']['enemy'],
                    'dmg_inflict_on_ally': infos[i]['red_stats']['damage_inflicted_on']['ally'],
                    'dmg_taken_by_ally': infos[i]['red_stats']['damage_taken_by']['ally'],
                    'dmg_taken_by_enemy': infos[i]['red_stats']['damage_taken_by']['enemy'],
                    })
             
        return obs, rewards, dones, infos

    def close(self) -> None:
        return self.venv.close()

if __name__ == '__main__':  
    args = cfg.args


    stats_dir = './runs/stats_{}'.format(args.logdir)
    env_params = {"training_tanks": [0], "static_tanks": [1, 2,3,4,5,6,7,8,9], "random_tanks": [], "disable_shooting": [1, 2,3,4,5,6,7,8,9],
                "friendly_fire":True, 'kill_bonus':False, 'death_penalty':False, 'take_damage_penalty': False,
                'tblogs':stats_dir, 'penalty_weight':args.penalty_weight, 'reward_weight':1.0, 'log_statistics': True, 'timeout': args.timeout}
    if args.record_stacked:
        print('load path', args.save_path)
        #env = DummyVecEnv([lambda: make_env_stacked(**env_params)])
        env = make_env_stacked(**env_params)
        if args.save_path is not None:
            model = PPO.load(args.save_path)
        else:
            policy_kwargs = {}
            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=512),
                net_arch=[dict(pi=[512], vf=[512])]
            )
            model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, n_steps=args.n_steps, 
                    verbose=2, batch_size=32, ent_coef=args.ent_coef)
        observation = env.reset()
        step = 0
        old_step =0
        env_count = 0
        game = 0
        observation_list = []
        while game < args.n_episode:
            action, _ = model.predict(observation)
            #action = np.random.rand(5,3)
            observation, reward, done, info = env.step(action)
            img = env.overview_map()
            observation_list.append(img)
            if done:
                step = 0
                game += 1
                observation = env.reset()
        out = cv2.VideoWriter(args.video_path, cv2.VideoWriter_fourcc(*"MJPG"), 5, (128, 128), True)
        for img in observation_list:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img)
        out.release()

    elif args.record_rgb:
        print('load path', args.save_path)
        if args.save_path is not None:
            model = PPO.load(args.save_path)
        env = DummyVecEnv([lambda: make_env_rgb(**env_params)])
        if args.stack_frame > 0:
            env = VecFrameStack(env, 4)
        #model = PPO("CnnPolicy", env, n_steps=args.horizon, verbose=2)
        #env = create_env()
        observation = env.reset()
        step = 0
        old_step =0
        env_count = 0
        game = 0
        observation_list = []
        while game < args.n_episode:
            action, _ = model.predict(observation)
            #action = np.random.rand(5,3)
            observation, reward, done, info = env.step(action)
            img = env.overview_map()
            observation_list.append(observation)
            if done:
                step = 0
                game += 1
                observation = env.reset()
        out = cv2.VideoWriter(args.video_path, cv2.VideoWriter_fourcc(*"MJPG"), 5, (128, 128), True)
        for img in observation_list:
            img = np.squeeze(img, axis=0)[:,:,-3:]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img)
        out.release()

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

    else:#training
        date_str = datetime.now().strftime("%y-%m-%d-%H:%M:%S")
        if args.testing:
            save_path = './testing/'+date_str+'-'+args.desc
        else:
            save_path = './results/'+date_str+'-'+args.desc
        os.mkdir(save_path)
        with open(save_path+'/config.yaml', 'w') as file:
            yaml.dump(args.__dict__,file)


        def create_env_rgb():
            #return Monitor(make_env(**env_params))
            return make_env_rgb(**env_params)
        def create_env_stacked():
            return make_env_stacked(**env_params)

        if args.env_stacked:
            create_env = create_env_stacked
        elif args.env_rgb:
            create_env = create_env_rgb

        if args.n_env == 1:
            env = create_env()
            check_env(env)
        else:
            env = SubprocVecEnv([create_env] * args.n_env)
            #env = DummyVecEnv([create_env] * args.n_env)
            #if args.stack_frame > 0:
            #    env = VecFrameStack(env, 4)

        env = VecMonitor(env)
        env = CustomMonitor(env)
        
        if args.save_path is not None:
            model = PPO.load(args.save_path, env=env)
        else:
            policy_kwargs = {}
            if args.env_stacked or args.model_size == 'large':
                policy_kwargs = dict(
                    features_extractor_class=CustomCNN,
                    features_extractor_kwargs=dict(features_dim=512),
                    net_arch=[dict(pi=[512], vf=[512])]
                )

            model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, n_steps=args.n_steps, 
                    verbose=2, batch_size=32, ent_coef=args.ent_coef,# clip_range=0.1, n_epochs=4,
                    tensorboard_log=save_path)


        checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=save_path + '/checkpoints', name_prefix='rl_model')
        tensorboard_callback = TensorboardCallback()
        callback_list = [checkpoint_callback, tensorboard_callback]
        model.learn(total_timesteps=args.timestep, callback=callback_list)

