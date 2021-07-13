# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
# To create a script for the arena, we need to interact with a UserStem object.
# This object is active only on the root process, and broadcasts commands to the
# other processes.

import math, time, random
import os

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *
import argparse
import json
import shutil

from algos.improved_ppo.ppo import PPOPolicy as ImprovedPPOPolicy
import my_config as cfg

args = cfg.args

additional_policies = {'improved_ppo': ImprovedPPOPolicy}
arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES, additional_policies)

# --- only the root process will get beyond this point ---

# the first 5 players in the gamew will use policy 1
match_list = [[1,1,1,1,1]]

# policy 1 is PPO
policy_types = {1:"improved_ppo"}

stats_dir = './runs/stats_{}'.format(args.logdir)
if os.path.exists(stats_dir):
    shutil.rmtree(stats_dir)
os.makedirs(stats_dir, exist_ok=True)

#kwargs to configure the environment
kwargs_1 = {"static_tanks":[], "random_tanks":[5,6,7,8,9], "disable_shooting":[],
            "friendly_fire":args.friendly_fire, 'kill_bonus':args.kill_bonus,
            'death_penalty':args.death_penalty, 'tblogs':stats_dir,
            'penalty_weight':args.penalty_weight, 'reward_weight':args.reward_weight}

if args.eval_mode:
    kwargs_2 = {1: {'eval_mode': True, 'external_saved_file': './logs/curriculum-deneme/policy_1/ppo_save_8640000.zip'}}
else:
    kwargs_2 = {1: {'optim_stepsize': args.optim_stepsize, 'optim_batchsize': args.optim_batchsize,
                    'schedule': args.schedule, 'policy_type': args.policy_type,
                    'timesteps_per_actorbatch': args.horizon}}

with open(os.path.join(stats_dir, 'policy_params.json'), 'w+') as f:
    json.dump(kwargs_2, f)

# run each copy of the environment for 300k steps
arena.kickoff(match_list, policy_types, 300000, scale=True, render=False, env_kwargs=kwargs_1, policy_kwargs=kwargs_2)
