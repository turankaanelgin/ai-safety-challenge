# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
# To create a script for the arena, we need to interact with a UserStem object.
# This object is active only on the root process, and broadcasts commands to the
# other processes.

import math, time, random
import os
import sys

import numpy as np

sys.path.append('spinningup')

from stems import *
from arena5.core.policy_record import *
import itertools

from algos.torch_ppo.ppo import PPOPolicy as TorchPPOPolicy
from algos.torch_ppo.core import MLPActorCritic
import my_config as cfg


args = cfg.args
args_dict = cfg.args_dict

additional_policies = {'torch_ppo': TorchPPOPolicy}
if args.seed != -1:
    if len(args.seed) == 1:
        env_seeds = [args.seed]
    else:
        env_seeds = args.seed
else:
    env_seeds = []
arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES, additional_policies)

# --- only the root process will get beyond this point ---
# the first 5 players in the gamew will use policy 1

grid = itertools.product(args_dict['reward_weight'],
                         args_dict['penalty_weight'],
                         args_dict['ff_weight'],
                         args_dict['policy_lr'],
                         args_dict['value_lr'],
                         args_dict['policy_lr_schedule'],
                         args_dict['value_lr_schedule'],
                         args_dict['steps_per_epoch'])
grid = [{'reward_weight': x[0],
         'penalty_weight': x[1],
         'ff_weight': x[2],
         'policy_lr': x[3],
         'value_lr': x[4],
         'policy_lr_schedule': x[5],
         'value_lr_schedule': x[6],
         'steps_per_epoch': x[7],
         'death_penalty': False,
         'friendly_fire': True,
         'kill_bonus': False,
         'take_damage_penalty': True} for x in grid]

print('Total number of configurations:', len(grid))

match_list = [[i,i,i,i,i] for i in range(1, len(grid)+1)]
policy_types = {}
for i in range(1, len(grid)+1):
    policy_types[i] = 'torch_ppo'
print('MATCH LIST:', match_list)

policy_folder_names = []
for config in grid:
    policy_folder_names.append('lrp={}{}__lrv={}{}__r={}__p={}__ff={}__H={}'.format(config['policy_lr'],
                                                                                 config['policy_lr_schedule'],
                                                                                 config['value_lr'],
                                                                                 config['value_lr_schedule'],
                                                                                 config['reward_weight'],
                                                                                 config['penalty_weight'],
                                                                                 config['ff_weight'],
                                                                                 config['steps_per_epoch']))
print('POLICY FOLDER NAMES:', policy_folder_names)

stats_dir = './runs/stats_{}'.format(args.logdir)
os.makedirs(stats_dir, exist_ok=True)

#kwargs to configure the environment
if args.eval_mode:
    kwargs_1 = {"static_tanks": [], "random_tanks": [5, 6, 7, 8, 9], "disable_shooting": [],
                "friendly_fire":False, 'kill_bonus':False, 'death_penalty':False, 'take_damage_penalty': True,
                'tblogs':stats_dir, 'penalty_weight':1.0, 'reward_weight':1.0, 'log_statistics': True, 'timeout': 500}
else:
    kwargs_1 = []
    for config in grid:
        kwargs_1.append({'static_tanks': [], 'random_tanks': [5,6,7,8,9], 'disable_shooting': [],
                         'friendly_fire': True, 'kill_bonus': False, 'death_penalty': False,
                         'take_damage_penalty': True, 'tblogs': stats_dir,
                         'penalty_weight': config['penalty_weight'], 'reward_weight': config['reward_weight'],
                         'friendly_fire_weight': config['ff_weight'], 'timeout': 500, 'seed': args.seed})
    if len(kwargs_1) == 1:
        kwargs_1 = kwargs_1[0]

if args.eval_mode:
    kwargs_2 = {1: {'eval_mode': True, 'model_path': './models/iter-4-for-real2/75000.pth'},
                2: {'eval_mode': True, 'model_path': './models/no-penalty-but-friendly/100000.pth'},
                3: {'eval_mode': True, 'model_path': './models/friendly-fire-0.8-replicated-v3/4000000.pth'},
                4: {'eval_mode': True, 'model_path': './models/frozen-cnn-0.8/4000000.pth'},
                5: {'eval_mode': True, 'model_path': './models/frozen-cnn-0.9/4000000.pth'},
                6: {'eval_mode': True, 'model_path': './models/frozen-cnn-0.8/4000000.pth'},
                8: {'eval_mode': True, 'model_path': './models/frozen-cnn-0.9/4000000.pth'}}
else:
    kwargs_2 = {}
    for idx, config in enumerate(grid):
        kwargs_2[idx+1] = {'steps_per_epoch': config['steps_per_epoch'], 'train_pi_iters': 4, 'train_v_iters': 4,
                         'actor_critic': MLPActorCritic, 'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0,
                         'model_id': 'iter-4-deneme-{}'.format(idx), 'model_path': './models/frozen-cnn-0.8/4000000.pth',
                         'pi_lr': config['policy_lr'], 'vf_lr': config['value_lr'], 'pi_scheduler': config['policy_lr_schedule'],
                         'vf_scheduler': config['value_lr_schedule']}

# run each copy of the environment for 300k steps
arena.kickoff(match_list, policy_types, args.num_iter, scale=True, render=False, env_kwargs=kwargs_1, policy_kwargs=kwargs_2,
              policy_folder_names=policy_folder_names, env_seeds=env_seeds)
