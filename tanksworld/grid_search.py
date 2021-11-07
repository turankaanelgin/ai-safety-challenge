import subprocess
import argparse
import os
import json
import numpy as np

import my_config as cfg

_MAX_INT = 2147483647 #Max int for Unity ML Seed


additional_args = cfg.args_dict
n_env_seeds = additional_args['n_env_seeds']
n_policy_seeds = additional_args['n_policy_seeds']
num_iter = additional_args['num_iter']

env_seed_arg = additional_args['env_seed']
if isinstance(env_seed_arg, list):
    env_seeds = env_seed_arg.copy()
else:
    env_seeds = [np.random.randint(_MAX_INT) for _ in range(n_env_seeds)]

policy_seed_arg = additional_args['policy_seed']
if isinstance(policy_seed_arg, list):
    policy_seeds = policy_seed_arg.copy()
else:
    policy_seeds = [np.random.randint(_MAX_INT) for _ in range(n_policy_seeds)]

del additional_args['n_env_seeds']
del additional_args['n_policy_seeds']
del additional_args['num_iter']
del additional_args['env_seed']
del additional_args['policy_seed']

if not os.path.exists(os.path.join('./logs', additional_args['logdir'])):
    os.mkdir(os.path.join('./logs', additional_args['logdir']))

init_seeds = os.path.join('./logs', additional_args['logdir'], 'seeds.json')
if os.path.exists(init_seeds):
    with open(init_seeds, 'r') as f:
        env_seeds = json.load(f)['env_seeds']

if os.path.exists(os.path.join('./logs', additional_args['logdir'], '{}.json'.format(additional_args['seed_id']))):
    with open(os.path.join('./logs', additional_args['logdir'], '{}.json'.format(additional_args['seed_id'])), 'r') as f:
        seed_list = json.load(f)
        env_seeds = seed_list['env_seeds']
        policy_seeds = seed_list['policy_seeds']
else:
    with open(os.path.join('./logs', additional_args['logdir'], '{}.json'.format(additional_args['seed_id'])), 'w+') as f:
        json.dump({'env_seeds': env_seeds, 'policy_seeds': policy_seeds}, f)

n_env_seeds = len(env_seeds)
n_policy_seeds = len(policy_seeds)
n = 1+2*len(cfg.grid)*n_policy_seeds

command = ['mpiexec', '-n', '{}'.format(n), 'python3.6', 'my_main_script.py']
for arg_name in additional_args:
    arg_value = additional_args[arg_name]
    if isinstance(arg_value, bool):
        if arg_value:
            command += ['--{}'.format(arg_name)]
    else:
        command += ['--{}'.format(arg_name)]
        if not isinstance(arg_value, list):
            command += ['{}'.format(arg_value)]
        else:
            for val in additional_args[arg_name]:
                command += ['{}'.format(val)]
command += ['--env_seed']
for s in env_seeds:
    command += ['{}'.format(s)]
command += ['--policy_seed']
for s in policy_seeds:
    command += ['{}'.format(s)]
command += ['--num_iter', '{}'.format(num_iter)]
print('COMMAND TO RUN:', ' '.join(command))
subprocess.run(command)