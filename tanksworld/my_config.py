# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import os
#from tanksworld.make_env import make_env
import gym
import os
import argparse
import itertools

parser = argparse.ArgumentParser(description='AI Safety TanksWorld')
parser.add_argument('--logdir',help='the location of saved policys and logs')
parser.add_argument('--exe', help='the absolute path of the tanksworld executable')
parser.add_argument('--teamname1', help='the name for team 1', default='red')
parser.add_argument('--teamname2', help='the name for team 2', default='blue')
parser.add_argument('--reward_weight', nargs='+', type=float, default=1.0)
parser.add_argument('--penalty_weight', nargs='+', type=float, default=1.0)
parser.add_argument('--ff_weight', nargs='+', type=float, default=0.0)
parser.add_argument('--curriculum_start', nargs='+', type=float, default=-1)
parser.add_argument('--curriculum_stop', nargs='+', type=float, default=-1)
parser.add_argument('--policy_lr', nargs='+', type=float, default=3e-4)
parser.add_argument('--value_lr', nargs='+', type=float, default=1e-3)
parser.add_argument('--policy_lr_schedule', nargs='+', type=str, default='cons')
parser.add_argument('--value_lr_schedule', nargs='+', type=str, default='cons')
parser.add_argument('--steps_per_epoch', nargs='+', type=int, default=64)
parser.add_argument('--death_penalty', action='store_true', default=False)
parser.add_argument('--friendly_fire', action='store_true', default=True)
parser.add_argument('--kill_bonus', action='store_true', default=False)
parser.add_argument('--eval_mode', action='store_true', default=False)
parser.add_argument('--n_env_seeds', type=int, default=1)
parser.add_argument('--n_policy_seeds', type=int, default=1)
parser.add_argument('--num_iter', type=int, default=1000)
parser.add_argument('--env_seed', nargs='+', type=int, default=-1)
parser.add_argument('--policy_seed', nargs='+', type=int, default=-1)
parser.add_argument('--eval_checkpoint', type=str, default='')
args = parser.parse_args()
args_dict = vars(args)

for param in ['reward_weight', 'penalty_weight', 'ff_weight', 'policy_lr', 'value_lr',
              'policy_lr_schedule', 'value_lr_schedule', 'steps_per_epoch', 'curriculum_start',
              'curriculum_stop']:
    if type(args_dict[param]) != list:
        args_dict[param] = [args_dict[param]]

grid = itertools.product(args_dict['reward_weight'],
                         args_dict['penalty_weight'],
                         args_dict['ff_weight'],
                         args_dict['policy_lr'],
                         args_dict['value_lr'],
                         args_dict['policy_lr_schedule'],
                         args_dict['value_lr_schedule'],
                         args_dict['steps_per_epoch'],
                         args_dict['curriculum_start'],
                         args_dict['curriculum_stop'])
grid = [{'reward_weight': x[0],
         'penalty_weight': x[1],
         'ff_weight': x[2],
         'policy_lr': x[3],
         'value_lr': x[4],
         'policy_lr_schedule': x[5],
         'value_lr_schedule': x[6],
         'steps_per_epoch': x[7],
         'curriculum_start': x[8],
         'curriculum_stop': x[9],
         'death_penalty': False,
         'friendly_fire': True,
         'kill_bonus': False,
         'take_damage_penalty': True} for x in grid]

# Tell the arena where it can put log files that describe the results of
# specific policies.  This is also used to pass results between root processes.

LOG_COMMS_DIR = "logs/"+args.logdir+"/"

#os.makedirs(LOG_COMMS_DIR, exist_ok=True)

# Define where to find the environment
# if make_env is in this directory, os.getcwd() will suffice

MAKE_ENV_LOCATION = os.getcwd()

# Tell the arena what observation and action spaces to expect

#temp = make_env()

NUM_LIVE_TANKS = 5
OBS_SPACES = [gym.spaces.box.Box(0,255,(4,128,128))]*NUM_LIVE_TANKS
ACT_SPACES = [gym.spaces.box.Box(-1,1,(3,))]*NUM_LIVE_TANKS
