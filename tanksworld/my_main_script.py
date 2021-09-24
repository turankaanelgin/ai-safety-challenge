# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
# To create a script for the arena, we need to interact with a UserStem object.
# This object is active only on the root process, and broadcasts commands to the
# other processes.

import math, time, random
import os
import sys
sys.path.append('spinningup')

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *
import argparse
import json
import shutil

from algos.improved_ppo.ppo import PPOPolicy as ImprovedPPOPolicy
from algos.torch_ppo.ppo import PPOPolicy as TorchPPOPolicy
from algos.torch_sac.sac import SACPolicy as TorchSACPolicy
from algos.torch_ppo.core import MLPActorCritic
from algos.torch_ppo.core import MLPActorCriticBernoulli
from algos.torch_sac.core import MLPActorCritic as SACMLPActorCritic
from algos.random.random_policy import RandomPolicy
import my_config as cfg

args = cfg.args

additional_policies = {'improved_ppo': ImprovedPPOPolicy, 'torch_ppo': TorchPPOPolicy,
                       'random_policy': RandomPolicy}
arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES, additional_policies)

# --- only the root process will get beyond this point ---
# the first 5 players in the gamew will use policy 1

match_list = [[1,1,1,1,1]]
              #[2,2,2,2,2]]
              #[3,3,3,3,3]]
              #[4,4,4,4,4]]
              #[5,5,5,5,5]]
              #[6,6,6,6,6]]
              #[7,7,7,7,7]]

# policy 1 is PPO
policy_types = {1: 'torch_ppo', 2: 'torch_ppo', 3: 'torch_ppo',
                4: 'torch_ppo', 5: 'torch_ppo', 6: 'torch_ppo',
                7: 'torch_ppo'}

stats_dir = './runs/stats_{}'.format(args.logdir)
if os.path.exists(stats_dir):
    shutil.rmtree(stats_dir)
os.makedirs(stats_dir, exist_ok=True)

#kwargs to configure the environment
if args.eval_mode:
    kwargs_1 = {"static_tanks": [], "random_tanks": [5, 6, 7, 8, 9], "disable_shooting": [],
                "friendly_fire":False, 'kill_bonus':False, 'death_penalty':False, 'take_damage_penalty': True,
                'tblogs':stats_dir, 'penalty_weight':1.0, 'reward_weight':1.0, 'log_statistics': True, 'timeout': 500,
                'barrier_heuristic': False}
else:
    kwargs_1 = {'static_tanks': [], 'random_tanks': [5, 6, 7, 8, 9], 'disable_shooting': [],
                'friendly_fire': False, 'kill_bonus': False, 'death_penalty': False, 'take_damage_penalty': True,
                'tblogs': stats_dir, 'penalty_weight': 0.8, 'reward_weight': 1.0, 'timeout': 500,
                'log_statistics': True, 'no_timeout': False, 'barrier_heuristic': False}


if args.eval_mode:
    kwargs_2 = {#1: {'eval_mode': True, 'model_path': './models/beta-0.7/4250000.pth', 'actor_critic': MLPActorCriticBernoulli},
                1: {'eval_mode': True, 'model_path': './models/curriculum-0.5/4000000.pth'},
                2: {'eval_mode': True, 'model_path': './models/frozen-cnn-0.25/4000000.pth'},
                3: {'eval_mode': True, 'model_path': './models/frozen_cnn-0.7/3500000.pth'},
                4: {'eval_mode': True, 'model_path': './models/frozen-cnn-0.8/4000000.pth'},
                5: {'eval_mode': True, 'model_path': './models/frozen-cnn-0.9/4000000.pth'},
                6: {'eval_mode': True, 'model_path': './models/frozen-cnn-0.8/4000000.pth'},
                8: {'eval_mode': True, 'model_path': './models/frozen-cnn-0.9/4000000.pth'}}
else:
    kwargs_2 = {
                1: {'steps_per_epoch': 64, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 0.8, 'model_id': 'barrier-heuristic',
                    'model_path': './models/frozen-cnn-0.8/4000000.pth'},
                2: {'steps_per_epoch': 64, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 0.1, 'model_id': 'new-model-0.1-v2',
                    'model_path': './models/penalty-0.7/3750000.pth'},
                3: {'steps_per_epoch': 64, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 0.9, 'model_id': 'frozen-cnn-0.9',
                    'model_path': './models/penalty-0.7/3750000.pth'},
                4: {'steps_per_epoch': 64, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 0.1, 'model_id': 'frozen-cnn-0.1',
                    'model_path': './models/penalty-0.7/3750000.pth'},
                5: {'steps_per_epoch': 64, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 0.6, 'model_id': 'new-model-0.6-v2',
                    'model_path': './models/penalty-0.7/3750000.pth'},
                6: {'steps_per_epoch': 64, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0, 'model_id': 'frozen-cnn-1.0',
                    'model_path': './models/penalty-0.7/3750000.pth'},
                7: {'steps_per_epoch': 64, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0, 'model_id': 'frozen-cnn-1.0',
                    'model_path': './models/penalty-0.7/3750000.pth'}}

with open(os.path.join(stats_dir, 'policy_params.json'), 'w+') as f:
    temp_kwargs_2 = {}
    for policy_idx in kwargs_2:
        temp_kwargs_2[policy_idx] = {}
        for key in kwargs_2[policy_idx]:
            if key != 'actor_critic':
                temp_kwargs_2[policy_idx][key] = kwargs_2[policy_idx][key]
    json.dump(temp_kwargs_2, f)
with open(os.path.join(stats_dir, 'env_params.json'), 'w+') as f:
    json.dump(kwargs_1, f)

# run each copy of the environment for 300k steps
arena.kickoff(match_list, policy_types, 300000, scale=True, render=False, env_kwargs=kwargs_1, policy_kwargs=kwargs_2)
