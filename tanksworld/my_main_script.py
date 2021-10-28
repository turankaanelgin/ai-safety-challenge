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
from algos.torch_sac.core import MLPActorCritic as SACMLPActorCritic
from algos.random.random_policy import RandomPolicy
import my_config as cfg

args = cfg.args

additional_policies = {'improved_ppo': ImprovedPPOPolicy, 'torch_ppo': TorchPPOPolicy,
                       'random_policy': RandomPolicy}
arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES, additional_policies)

# --- only the root process will get beyond this point ---
# the first 5 players in the gamew will use policy 1

match_list = [#[1,1,1,1,1],
              #[2,2,2,2,2],
              #[3,3,3,3,3],
              #[4,4,4,4,4],
              #[5,5,5,5,5]]
              #[6,6,6,6,6],
              #[7,7,7,7,7],
               #[8,8,8,8,8]]
               [9,9,9,9,9],
               [10,10,10,10,10]]

# policy 1 is PPO
policy_types = {1: 'torch_ppo', 2: 'torch_ppo', 3: 'torch_ppo',
                4: 'torch_ppo', 5: 'torch_ppo', 6: 'torch_ppo',
                7: 'torch_ppo', 8: 'torch_ppo', 9: 'torch_ppo',
                10: 'torch_ppo'}

stats_dir = './runs/stats_{}'.format(args.logdir)
if os.path.exists(stats_dir):
    shutil.rmtree(stats_dir)
os.makedirs(stats_dir, exist_ok=True)

#kwargs to configure the environment
if args.eval_mode:
    kwargs_1 = {"static_tanks": [], "random_tanks": [5, 6, 7, 8, 9], "disable_shooting": [],
                "friendly_fire":False, 'kill_bonus':False, 'death_penalty':False, 'take_damage_penalty': True,
                'tblogs':stats_dir, 'penalty_weight':1.0, 'reward_weight':1.0, 'log_statistics': True, 'timeout': 500}
else:
    kwargs_1 = [{'static_tanks': [], 'random_tanks': [5, 6, 7, 8, 9], 'disable_shooting': [],
                'friendly_fire': False, 'kill_bonus': False, 'death_penalty': False, 'take_damage_penalty': True,
                'tblogs': stats_dir, 'penalty_weight': 0.6, 'reward_weight': 1.0, 'timeout': 500,
                'log_statistics': False, 'no_timeout': False, 'friendly_fire_weight': 0.6},
                {'static_tanks': [], 'random_tanks': [5, 6, 7, 8, 9], 'disable_shooting': [],
                 'friendly_fire': True, 'kill_bonus': False, 'death_penalty': False, 'take_damage_penalty': False,
                 'tblogs': stats_dir, 'penalty_weight': 0.6, 'reward_weight': 1.0, 'timeout': 500,
                 'log_statistics': False, 'no_timeout': False, 'friendly_fire_weight': 1.0},
                {'static_tanks': [], 'random_tanks': [5, 6, 7, 8, 9], 'disable_shooting': [],
                 'friendly_fire': True, 'kill_bonus': False, 'death_penalty': False, 'take_damage_penalty': True,
                 'tblogs': stats_dir, 'penalty_weight': 0.6, 'reward_weight': 1.0, 'timeout': 500,
                 'log_statistics': False, 'no_timeout': False, 'friendly_fire_weight': 0.6}]
    kwargs_1 = kwargs_1[0]


if args.eval_mode:
    kwargs_2 = {1: {'eval_mode': True, 'model_path': './models/friendly-fire-0.8-replicated/4000000.pth'},
                2: {'eval_mode': True, 'model_path': './models/no-penalty-but-friendly/100000.pth'},
                3: {'eval_mode': True, 'model_path': './models/friendly-fire-0.8-replicated-v3/4000000.pth'},
                4: {'eval_mode': True, 'model_path': './models/frozen-cnn-0.8/4000000.pth'},
                5: {'eval_mode': True, 'model_path': './models/frozen-cnn-0.9/4000000.pth'},
                6: {'eval_mode': True, 'model_path': './models/frozen-cnn-0.8/4000000.pth'},
                8: {'eval_mode': True, 'model_path': './models/frozen-cnn-0.9/4000000.pth'}}
else:
    kwargs_2 = {
                1: {'steps_per_epoch': 4, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0, 'model_id': 'iter-4-deneme1',
                    'model_path': './models/frozen-cnn-0.8/4000000.pth', 'pi_lr': 3e-4, 'vf_lr': 1e-3},
                2: {'steps_per_epoch': 4, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0, 'model_id': 'iter-4-deneme2',
                    'model_path': './models/frozen-cnn-0.8/4000000.pth', 'pi_lr': 5e-5, 'vf_lr': 1e-3},
                3: {'steps_per_epoch': 4, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0, 'model_id': 'iter-4-deneme3',
                    'model_path': './models/frozen-cnn-0.8/4000000.pth', 'pi_lr': 1e-5, 'vf_lr': 1e-3},
                4: {'steps_per_epoch': 4, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0, 'model_id': 'iter-4-for-real1',
                    'model_path': './models/frozen-cnn-0.8/4000000.pth', 'pi_lr': 3e-4, 'vf_lr': 5e-4},
                5: {'steps_per_epoch': 4, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0, 'model_id': 'iter-4-for-real2',
                    'model_path': './models/frozen-cnn-0.8/4000000.pth', 'pi_lr': 3e-4, 'vf_lr': 1e-4},
                6: {'steps_per_epoch': 4, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0, 'model_id': 'iter-4-deneme6',
                    'model_path': './models/frozen-cnn-0.8/4000000.pth', 'pi_lr': 5e-5, 'vf_lr': 5e-4},
                7: {'steps_per_epoch': 4, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0, 'model_id': 'iter-4-deneme7',
                    'model_path': './models/frozen-cnn-0.8/4000000.pth', 'pi_lr': 5e-5, 'vf_lr': 1e-4},
                8: {'steps_per_epoch': 4, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0, 'model_id': 'iter-4-deneme8',
                    'model_path': './models/frozen-cnn-0.8/4000000.pth', 'pi_lr': 3e-4, 'vf_lr': 1e-3, 'schedule': 'linear'},
                9: {'steps_per_epoch': 4, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0, 'model_id': 'iter-4-deneme9',
                    'model_path': './models/frozen-cnn-0.8/4000000.pth', 'pi_lr': 3e-4, 'vf_lr': 1e-3, 'schedule': 'smart'},
                10: {'steps_per_epoch': 4, 'train_pi_iters': 4, 'train_v_iters': 4, 'actor_critic': MLPActorCritic,
                    'ac_kwargs': {'hidden_sizes': (64, 64)}, 'neg_weight_constant': 1.0, 'model_id': 'iter-4-deneme10',
                    'model_path': './models/frozen-cnn-0.8/4000000.pth', 'pi_lr': 3e-4, 'vf_lr': 1e-3, 'schedule': 'sqrt'}}

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
