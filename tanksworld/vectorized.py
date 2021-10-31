from arena5.core.env_process import EnvironmentProcess
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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

if __name__ == '__main__':  
    args = cfg.args

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
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
                nn.ReLU(),
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


    stats_dir = './runs/stats_{}'.format(args.logdir)
    kwargs_1 = {"static_tanks": [], "random_tanks": [5, 6, 7, 8, 9], "disable_shooting": [],
                "friendly_fire":False, 'kill_bonus':False, 'death_penalty':False, 'take_damage_penalty': True,
                'tblogs':stats_dir, 'penalty_weight':1.0, 'reward_weight':1.0, 'log_statistics': True, 'timeout': 500}
    def create_env():
        #return Monitor(make_env(**kwargs_1))
        return make_env(**kwargs_1)
    if args.record:
        model = PPO.load(args.save_path)
        env = create_env()
        observation = env.reset()
        step = 0
        old_step =0
        env_count = 0
        game = 0
        while game < 500:
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(action)
            if done:
                game += 1
                observation = env.reset()

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

    else:
        from datetime import datetime
        date_str = datetime.now().strftime("%y-%m-%d-%H:%M")
        if args.testing:
            save_path = './testing/'+date_str+'-'+args.desc
        else:
            save_path = './results/'+date_str+'-'+args.desc
        import os
        os.mkdir(save_path)
        import yaml
        with open(save_path+'/config.yaml', 'w') as file:
            yaml.dump(args.__dict__,file)


        if args.n_env == 1:
            env = create_env()
        else:
            env = SubprocVecEnv([create_env] * args.n_env)
            env = VecMonitor(env)

        #env = Monitor(env)
        #print(is_wrapped(env, Monitor))
        #import pdb; pdb.set_trace();
        #model = CustomCNN(env.observation_space)
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[512, dict(pi=[512, 512], vf=[512, 512])]
        )
        #model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs)1024


        
        checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=save_path + '/checkpoints', name_prefix='rl_model')
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, n_steps=args.horizon, verbose=2,tensorboard_log=save_path)
        model.learn(total_timesteps=args.timestep, callback=checkpoint_callback)


