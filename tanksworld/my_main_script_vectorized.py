import os
import pdb
import argparse

import numpy as np
import torch as th
import torch.nn as nn
import gym

from algos.torch_ppo.mappo_gpu import PPOPolicy as TorchGPUMAPPOPolicy
from algos.torch_ppo.mappo_gpu_separate_env import PPOPolicy as TorchGPUMAPPOPolicyUpdated
from algos.fast_ppo.ppo import PPO as FastPPO
from algos.fast_ppo.policies import MyActorCriticCnnPolicy
from algos.fast_ppo.vec_env import VecMonitor, SubprocVecEnv

from stable_baselines3.common.policies import register_policy
from make_env import make_env

#from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback

from core.policy_record import PolicyRecord


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir',help='the location of saved policys and logs')
    parser.add_argument('--exe', help='the absolute path of the tanksworld executable')
    parser.add_argument('--teamname1', help='the name for team 1', default='red')
    parser.add_argument('--teamname2', help='the name for team 2', default='blue')
    parser.add_argument('--reward_weight', type=float, default=1.0)
    parser.add_argument('--penalty_weight', type=float, default=1.0)
    parser.add_argument('--ff_weight', type=float, default=0.0)
    parser.add_argument('--curriculum_start', type=float, default=-1)
    parser.add_argument('--curriculum_stop', type=float, default=-1)
    parser.add_argument('--policy_lr', type=float, default=3e-4)
    parser.add_argument('--value_lr', type=float, default=1e-3)
    parser.add_argument('--policy_lr_schedule', type=str, default='cons')
    parser.add_argument('--value_lr_schedule', type=str, default='cons')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--death_penalty', action='store_true', default=False)
    parser.add_argument('--friendly_fire', action='store_true', default=True)
    parser.add_argument('--take_damage_penalty', action='store_true', default=True)
    parser.add_argument('--kill_bonus', action='store_true', default=False)
    parser.add_argument('--eval_mode', action='store_true', default=False)
    parser.add_argument('--n_env_seeds', type=int, default=1)
    parser.add_argument('--n_policy_seeds', type=int, default=1)
    parser.add_argument('--num_iter', type=int, default=1000)
    parser.add_argument('--env_seed', nargs='+', type=int, default=-1)
    parser.add_argument('--policy_seed', type=int, default=-1)
    parser.add_argument('--eval_checkpoint', type=str, default='')
    parser.add_argument('--save_tag', type=str, default='')
    parser.add_argument('--load_from_checkpoint', action='store_true', default=False)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--seed_index', type=int, default=0)
    parser.add_argument('--use_popart', action='store_true', default=False)
    parser.add_argument('--use_rnn', action='store_true', default=False)
    parser.add_argument('--freeze_rep', action='store_false', default=True)
    args = parser.parse_args()


    class CustomCNN(BaseFeaturesExtractor):
        """
        CNN from DQN nature paper:
            Mnih, Volodymyr, et al.
            "Human-level control through deep reinforcement learning."
            Nature 518.7540 (2015): 529-533.
        :param observation_space:
        :param features_dim: Number of features extracted.
            This corresponds to the number of unit for the last layer.
        """

        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
            super(CustomCNN, self).__init__(observation_space, features_dim)

            n_input_channels = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

        def forward(self, observations: th.Tensor) -> th.Tensor:
            if len(observations.shape) == 6:
                observations = observations.squeeze(2)
            observations = observations.reshape(observations.shape[0]*observations.shape[1],
                                                observations.shape[2], observations.shape[3],
                                                observations.shape[4])
            return self.cnn(observations)


    config = vars(args)
    folder_name = 'pi_lr={}{}__vf_lr={}{}__p={}__ff={}__B={}'.format(config['policy_lr'],
                                                                config['policy_lr_schedule'],
                                                                config['value_lr'],
                                                                config['value_lr_schedule'],
                                                                config['penalty_weight'],
                                                                config['ff_weight'],
                                                                config['batch_size'])
    if config['curriculum_stop'] >= 0.0:
        folder_name += '__CS={}__CF={}'.format(config['penalty_weight'], config['curriculum_stop'])
    if config['use_popart']:
        folder_name += '__popart'
    if config['use_rnn']:
        folder_name += '__rnn'
    folder_name += '__{}'.format(args.save_tag)
    folder_name += '__seed{}'.format(config['seed_index'])

    env_seed = args.env_seed
    stats_dir = './runs/stats_{}'.format(args.logdir)
    os.makedirs(stats_dir, exist_ok=True)

    kwargs_1 = []
    for seed in env_seed:
        kwargs_1.append({'exe': args.exe,
                         'static_tanks': [], 'random_tanks': [5,6,7,8,9], 'disable_shooting': [],
                         'friendly_fire': True, 'kill_bonus': False, 'death_penalty': False,
                         'take_damage_penalty': True, 'tblogs': stats_dir,
                         'penalty_weight': config['penalty_weight'], 'reward_weight': config['reward_weight'],
                         'friendly_fire_weight': config['ff_weight'], 'timeout': 500, 'log_statistics': True,
                         'seed': seed, 'curriculum_stop': config['curriculum_stop'], 'curriculum_steps': args.num_iter})

    if len(env_seed) == 1:
        env = make_env(**kwargs_1[0])
    else:
        env_functions = []
        for i in range(len(env_seed)):
            env_functions.append(lambda : make_env(**kwargs_1[i]))
        stacked_env = [env_functions[i] for i in range(len(env_seed))]
        env = SubprocVecEnv(stacked_env)

    pr = PolicyRecord(1, folder_name, './logs/'+args.logdir+'/')
    '''
    policy_kwargs = {
        'policy': 'MyCnnPolicy',
        'env': env,
        'learning_rate': config['policy_lr'],
        'n_steps': config['n_steps'],
        'batch_size': config['batch_size'],
        'n_epochs': config['n_epochs'],
        'gamma': 0.99,
        'gae_lambda': 0.97,
        'clip_range': 0.2,
        'ent_coef': config['ent_coef'],
        'vf_coef': config['vf_coef'],
        'max_grad_norm': float('inf'),
        'use_sde': False,
        'target_kl': 0.01,
        'seed': args.policy_seed,
        'policy_kwargs': {'features_extractor_class': CustomCNN},
    }
    '''
    model_id = '{}---{}-{}-{}'.format(args.logdir.split('/')[-1], folder_name, args.seed_index, args.save_tag)
    model_path = None
    if args.load_from_checkpoint:
        model_path = os.path.join(pr.data_dir, 'checkpoints', model_id)
        checkpoint_files = os.listdir(model_path)
        checkpoint_files.sort(key=lambda f: int(f.split('.')[0]))
        model_path = os.path.join(model_path, checkpoint_files[-1])

    policy_kwargs_old = {
        'steps_per_epoch': config['batch_size'],
        'train_pi_iters': 4,
        'train_v_iters': 4,
        'pi_lr': config['policy_lr'],
        'vf_lr': config['value_lr'],
        'ent_coef': config['ent_coef'],
        'pi_scheduler': config['policy_lr_schedule'],
        'vf_scheduler': config['value_lr_schedule'],
        'seed': args.policy_seed,
        'cnn_model_path': './models/frozen-cnn-0.8/4000000.pth',
        'model_path': model_path,
        'n_envs': len(env_seed),
        'model_id': model_id,
        'save_dir': os.path.join(pr.data_dir, 'checkpoints'),
        'freeze_rep': config['freeze_rep'],
        'use_rnn': config['use_rnn'],
        'num_states': 4,
        'use_popart': config['use_popart'],
    }

    '''
    register_policy('MyCnnPolicy', MyActorCriticCnnPolicy)
    policy = FastPPO(**policy_kwargs)
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=pr.data_dir, name_prefix='tanks_model')
    policy.learn(total_timesteps=args.num_iter, callback=None, policy_record=pr)
    '''

    #policy = TorchGPUMAPPOPolicyUpdated(env, False, **policy_kwargs_old)
    policy = TorchGPUMAPPOPolicy(env, False, **policy_kwargs_old)
    policy.run(pr, num_steps=args.num_iter)