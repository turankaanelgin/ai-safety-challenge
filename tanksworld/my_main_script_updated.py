import math, time, random
import os
import sys
import argparse

import numpy as np

sys.path.append('spinningup')

from algos.torch_ppo.ppo import PPOPolicy as TorchPPOPolicy
from algos.torch_ppo.mappo import PPOPolicy as TorchMAPPOPolicy
from algos.torch_ppo.mappo_gpu import PPOPolicy as TorchGPUMAPPOPolicy
from algos.torch_ppo.core import MLPActorCritic
from make_env import make_env
from env_wrappers import ShareSubprocVecEnv

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

from arena5.core.policy_record import PolicyRecord


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
    parser.add_argument('--steps_per_epoch', type=int, default=64)
    parser.add_argument('--death_penalty', action='store_true', default=False)
    parser.add_argument('--friendly_fire', action='store_true', default=True)
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
    parser.add_argument('--seed_index', type=int, default=0)
    args = parser.parse_args()

    config = vars(args)
    folder_name = 'lrp={}{}__lrv={}{}__r={}__p={}__ff={}__H={}__{}'.format(config['policy_lr'],
                                                                         config['policy_lr_schedule'],
                                                                         config['value_lr'],
                                                                         config['value_lr_schedule'],
                                                                         config['reward_weight'],
                                                                         config['penalty_weight'],
                                                                         config['ff_weight'],
                                                                         config['steps_per_epoch'],
                                                                         args.save_tag)
    if config['curriculum_start'] >= 0.0:
        folder_name += '__CS={}__CF={}'.format(config['curriculum_start'], config['curriculum_stop'])
    folder_name += '__seed{}'.format(config['seed_index'])

    if isinstance(args.env_seed, int):
        env_seed = [args.env_seed]
    else:
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
                         'seed': seed})

    model_id = 'model'+args.save_tag
    model_path = None

    kwargs_2 = {'steps_per_epoch': config['steps_per_epoch'], 'train_pi_iters': 4, 'train_v_iters': 4,
                'actor_critic': MLPActorCritic, 'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0,
                'model_id': model_id, 'cnn_model_path': './models/frozen-cnn-0.8/4000000.pth', 'model_path': model_path,
                'pi_lr': config['policy_lr'], 'vf_lr': config['value_lr'], 'pi_scheduler': config['policy_lr_schedule'],
                'vf_scheduler': config['value_lr_schedule'], 'seed': args.policy_seed,
                'curriculum_start': config['curriculum_start'], 'curriculum_stop': config['curriculum_stop'],
                'save_dir': os.path.join('./logs', args.logdir, folder_name), 'n_envs': len(env_seed)}

    if len(env_seed) == 1:
        env = make_env(**kwargs_1[0])
    else:
        env_functions = []
        for i in range(len(args.env_seed)):
            env_functions.append(lambda : make_env(**kwargs_1[i]))
        stacked_env = [env_functions[i] for i in range(len(args.env_seed))]
        env = SubprocVecEnv(stacked_env)

    pr = PolicyRecord(1, folder_name, './logs/'+args.logdir+'/')
    policy = TorchGPUMAPPOPolicy(env, eval_mode=False, **kwargs_2)
    policy.run(pr, num_steps=args.num_iter)