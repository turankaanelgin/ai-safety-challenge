# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
# To create a script for the arena, we need to interact with a UserStem object.
# This object is active only on the root process, and broadcasts commands to the
# other processes.

import math, time, random
import os
import sys

import numpy as np

sys.path.append('spinningup')

from core.stems import *
from arena5.core.policy_record import *
import itertools

from algos.torch_ppo.ppo import PPOPolicy as TorchPPOPolicy
from algos.torch_ppo.mappo import PPOPolicy as TorchMAPPOPolicy
from algos.torch_ppo.core import MLPActorCritic
import my_config as cfg


args = cfg.args
args_dict = cfg.args_dict

additional_policies = {'torch_ppo': TorchPPOPolicy, 'torch_mappo': TorchMAPPOPolicy}
if args.env_seed != -1:
    if len(args.env_seed) == 1:
        env_seeds = [args.env_seed]
    else:
        env_seeds = args.env_seed
else:
    env_seeds = []
if args.policy_seed != -1:
    if len(args.policy_seed) == 1:
        policy_seeds = [args.policy_seed]
    else:
        policy_seeds = args.policy_seed
else:
    policy_seeds = []
arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES, additional_policies)

# --- only the root process will get beyond this point ---
# the first 5 players in the gamew will use policy 1
grid = cfg.grid
print('Total number of configurations:', len(grid))

if args.eval_mode:
    match_list = [[1,1,1,1,1]]
else:
    match_list = [[[i,i,i,i,i]] for i in range(1, len(grid)*len(policy_seeds)+1)]
policy_types = {i: 'torch_mappo' for i in range(1, len(grid)*len(policy_seeds)+1)}
print('MATCH LIST:', match_list)

colors = [(1.0,0,0), (0,0,1.0), (0,1.0,0), (1.0,1.0,0), (0,1.0,1.0), (1.0,0,1.0)]
assert len(colors) >= len(policy_seeds)

if args.eval_mode:
    policy_folder_names = [args.eval_checkpoint.split('/')[-2]]
else:
    policy_folder_names = []
    for config in grid:
        folder_name = 'lrp={}{}__lrv={}{}__r={}__p={}__ff={}__H={}'.format(config['policy_lr'],
                                                                                 config['policy_lr_schedule'],
                                                                                 config['value_lr'],
                                                                                 config['value_lr_schedule'],
                                                                                 config['reward_weight'],
                                                                                 config['penalty_weight'],
                                                                                 config['ff_weight'],
                                                                                 config['steps_per_epoch'])
        if config['curriculum_start'] >= 0.0:
            folder_name += '__CS={}__CF={}'.format(config['curriculum_start'], config['curriculum_stop'])
        policy_folder_names += [folder_name] * len(policy_seeds)

plot_colors = colors[:len(policy_seeds)]*len(grid)
plot_colors = {i+1: plot_colors[i] for i in range(len(plot_colors))}

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
        for _ in policy_seeds:
            kwargs_1.append({'static_tanks': [], 'random_tanks': [5,6,7,8,9], 'disable_shooting': [],
                         'friendly_fire': True, 'kill_bonus': False, 'death_penalty': False,
                         'take_damage_penalty': True, 'tblogs': stats_dir,
                         'penalty_weight': config['penalty_weight'], 'reward_weight': config['reward_weight'],
                         'friendly_fire_weight': config['ff_weight'], 'timeout': 500, 'log_statistics': True})
    if len(kwargs_1) == 1:
        kwargs_1 = kwargs_1[0]

if args.eval_mode:
    kwargs_2 = {1: {'eval_mode': True, 'model_path': args.eval_checkpoint}}
else:
    kwargs_2 = {}
    idx = 0
    for policy_idx, config in enumerate(grid):
        for seed_idx, seed in enumerate(policy_seeds):
            kwargs_2[idx+1] = {'steps_per_epoch': config['steps_per_epoch'], 'train_pi_iters': 4, 'train_v_iters': 4,
                         'actor_critic': MLPActorCritic, 'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0,
                         'model_id': '{}-{}'.format(policy_folder_names[policy_idx], seed_idx),
                         'model_path': './models/frozen-cnn-0.8/4000000.pth',
                         'pi_lr': config['policy_lr'], 'vf_lr': config['value_lr'], 'pi_scheduler': config['policy_lr_schedule'],
                         'vf_scheduler': config['value_lr_schedule'], 'seed': seed, 'curriculum_start': config['curriculum_start'],
                         'curriculum_stop': config['curriculum_stop']}
            idx += 1

# run each copy of the environment for 300k steps
arena.kickoff(match_list, policy_types, args.num_iter, scale=True, render=False, env_kwargs=kwargs_1, policy_kwargs=kwargs_2,
              policy_folder_names=policy_folder_names, env_seeds=env_seeds, plot_colors=plot_colors)
