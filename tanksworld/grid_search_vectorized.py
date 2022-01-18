import pdb
import subprocess
from subprocess import Popen
import argparse
import os
import json
import numpy as np
import pprint

import my_config as cfg

_MAX_INT = 2147483647 #Max int for Unity ML Seed


args = cfg.args
args_dict = cfg.args_dict
n_env_seeds = args.n_env_seeds
n_policy_seeds = args.n_policy_seeds
num_iter = args.num_iter

env_seed_arg = args.env_seed
if isinstance(env_seed_arg, list):
    env_seeds = env_seed_arg.copy()
else:
    env_seeds = [np.random.randint(_MAX_INT) for _ in range(n_env_seeds)]

policy_seed_arg = args.policy_seed
if isinstance(policy_seed_arg, list):
    policy_seeds = policy_seed_arg.copy()
else:
    policy_seeds = [np.random.randint(_MAX_INT) for _ in range(n_policy_seeds)]

del args_dict['n_env_seeds']
del args_dict['n_policy_seeds']
del args_dict['num_iter']
del args_dict['env_seed']
del args_dict['policy_seed']

if not os.path.exists(os.path.join('./logs', args.logdir)):
    os.mkdir(os.path.join('./logs', args.logdir))

init_seeds = os.path.join('./logs', args.logdir, 'seeds.json')
if os.path.exists(init_seeds):
    with open(init_seeds, 'r') as f:
        env_seeds = json.load(f)['env_seeds']

if os.path.exists(os.path.join('./logs', args.logdir, 'seeds.json')):
    with open(os.path.join('./logs', args.logdir, 'seeds.json'), 'r') as f:
        seed_list = json.load(f)
        env_seeds = seed_list['env_seeds']
        policy_seeds = seed_list['policy_seeds']
else:
    with open(os.path.join('./logs', args.logdir, 'seeds.json'), 'w+') as f:
        json.dump({'env_seeds': env_seeds, 'policy_seeds': policy_seeds}, f)

if n_env_seeds < len(env_seeds):
    env_seeds = env_seeds[:n_env_seeds]
if n_policy_seeds < len(policy_seeds):
    policy_seeds = policy_seeds[:n_policy_seeds]

commands = []
for config in cfg.grid:
    for e_seed_idx, e_seed in enumerate(env_seeds):
        for p_seed_idx, p_seed in enumerate(policy_seeds):
            command = ['python3.6', 'my_main_script_updated.py']
            command += ['--exe', args.exe]
            command += ['--logdir', args.logdir]
            for arg_name in config:
                arg_value = config[arg_name]
                if isinstance(arg_value, bool):
                    if arg_name in ['multiplayer', 'valuenorm', 'freeze_rep',
                                    'use_rnn', 'beta', 'fixed_kl', 'adaptive_kl',
                                    'eval_mode'] and arg_value:
                        command += ['--{}'.format(arg_name)]
                else:
                    command += ['--{}'.format(arg_name)]
                    command += ['{}'.format(arg_value)]
            command += ['--env_seed', '{}'.format(e_seed)]
            command += ['--policy_seed', '{}'.format(p_seed)]
            command += ['--seed_index', '{}'.format(e_seed_idx*len(policy_seeds)+p_seed_idx)]
            command += ['--num_iter', '{}'.format(num_iter)]
            if args.load_from_checkpoint:
                command += ['--load_from_checkpoint']
            commands.append(command)

for c in commands:
    print(' '.join(c))

procs = [ Popen(c) for c in commands ]
for p in procs:
   p.wait()