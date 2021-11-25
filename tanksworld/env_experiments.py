import gym
from gym import spaces
from arena5.core.env_process import EnvironmentProcess
from PIL import Image
from tanksworld.make_env import make_env
import numpy as np
from stable_baselines3 import PPO, DQN
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
from env_original.make_env import make_env as make_env_origin
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import rgb_env
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

class EnvRGB(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, env_name, width, height):
        super(EnvRGB, self).__init__()
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, height, width), dtype=np.uint8)
        #self.resize = T.Compose([T.ToPILImage(),
        #    T.Resize(40, interpolation=Image.CUBIC),
        #    T.ToTensor()])
        
    def resize(self, img):
        #img = img.transpose((2,0,1))
        pil_img = Image.fromarray(img, 'RGB')
        resize_img = pil_img.resize((self.width, self.height))
        resize_img = np.array(resize_img)
        resize_img = resize_img.transpose((2,0,1))
        return resize_img
    
    def step(self, action):
        obs, r, d, info = self.env.step(action)
        img = self.env.render(mode="rgb_array")
        img = self.resize(img)
        return img, r, d, info
    def reset(self):
        self.env.reset()
        img = self.env.render(mode="rgb_array")
        img = self.resize(img)
        return img
    def render(self, mode='human'):
        pass
    def close (self):
        self.env.close()

if __name__ == '__main__':  
    args = cfg.args
    def create_env1():
        env = EnvRGB('LunarLanderContinuous-v2', 60, 40)
        return env
    if args.record:
        env = create_env()
        observation = env.reset()
        step , old_step =0, 0
        env_count = 0
        game = 0
        episode_step = []
        while game < 500:
            step += 1
            observation, reward, done, info = env.step(list(np.random.rand(5,3)))
            if done:
                episode_step.append(step - old_step)
                np.save('tmp/origin', episode_step)
                old_step = step
                game += 1
                observation = env.reset()
    elif args.debug:
        env = gym.make("Pong-v0")
        model = PPO("CnnPolicy", env, n_steps=args.horizon, verbose=1)
        model.learn(total_timesteps=args.timestep)
    elif args.record_rgb:
        print('load path', args.save_path)
        #model = PPO.load(args.save_path)
        #env = make_env_rgb(**kwargs_1)
        env = create_env1()
        model = PPO("CnnPolicy", env)
        observation = env.reset()
        step = 0
        old_step =0
        env_count = 0
        game = 0
        observation_list = []
        while game < 1:
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(action)
            #print(98, 'step ======>', step)
            step += 1
            observation_list.append(observation)
            if done:
                step = 0
                np.save('tmp/rgb', observation_list)
                print(info['average'])
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                out = cv2.VideoWriter('tmp/videos/{}.mp4'.format(game),fourcc, 2, (128, 128))
                for img in observation_list:  
                    out.write(img) 
                out.release()
                observation_list = []
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

    elif args.testing:
        print("start testing training...")
        from datetime import datetime
        date_str = datetime.now().strftime("%y-%m-%d-%H:%M:%S")
        if args.testing:
            save_path = './testing/'+date_str+'-'+args.desc
        else:
            save_path = './results/'+date_str+'-'+args.desc
        import os
        os.mkdir(save_path)
        import yaml
        with open(save_path+'/config.yaml', 'w') as file:
            yaml.dump(args.__dict__,file)


        #if args.n_env == 1:
        #    env = create_env()
        #else:
        #    env = SubprocVecEnv([create_env] * args.n_env)
        #    env = VecMonitor(env)

        #env = Monitor(env)
        #print(is_wrapped(env, Monitor))
        #import pdb; pdb.set_trace();
        #model = CustomCNN(env.observation_space)
        #model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs)1024





        #env = EnvRGB('LunarLanderContinuous-v2',75, 50)
        #print(check_env(env))

        #policy_kwargs = dict(
        #    features_extractor_class=CustomCNN,
        #    features_extractor_kwargs=dict(features_dim=512),
        #    net_arch=[512, dict(pi=[512, 512], vf=[512, 512])]
        #)

        def create_env():
            env = gym.make('RGBCartPole-v0')
            env.seed(np.random.randint(10000))
            return env

        #env = SubprocVecEnv([create_env for i in range(args.n_env)])
        env = SubprocVecEnv([create_env for i in range(args.n_env)])
        #env = make_vec_env('RGBCartPole-v0', args.n_env)
        env = VecFrameStack(env, 4)
        env = VecMonitor(env)

        if args.save_path is not None:
            model = PPO.load(args.save_path)
            model.set_env(env)
        else:
            policy_kwargs = dict(
                net_arch=[dict(pi=[128], vf=[128])]
            )
        #    #model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, n_steps=args.horizon, verbose=2,tensorboard_log=save_path)
            model = PPO("CnnPolicy", env,  #policy_kwargs=policy_kwargs, \
                    n_steps=32, verbose=2,tensorboard_log=save_path,
                    batch_size=32, ent_coef=0.01, clip_range=0.1, n_epochs=4
                    )

        print(model.policy)
        checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=save_path + '/checkpoints', name_prefix='rl_model')
        model.learn(total_timesteps=args.timestep, callback=checkpoint_callback)


