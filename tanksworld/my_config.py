# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import os
#from tanksworld.make_env import make_env
import gym
import os
import argparse

parser = argparse.ArgumentParser(description='AI Safety TanksWorld')
parser.add_argument('--logdir',help='the location of saved policys and logs')
parser.add_argument('--exe', help='the absolute path of the tanksworld executable')
parser.add_argument('--desc') 
parser.add_argument('--teamname1', help='the name for team 1')
parser.add_argument('--teamname2', help='the name for team 2')
parser.add_argument('--reward_weight', type=float, default=1.0)
parser.add_argument('--penalty-weight', type=float, default=1.0)
parser.add_argument('--optim_stepsize', type=float, default=0.001)
parser.add_argument('--ent-coef', type=float, default=0.0)
parser.add_argument('--optim_batchsize', type=int, default=64)
parser.add_argument('--n-steps', type=int, default=1024)
parser.add_argument('--timestep', type=int, default=1000000)
parser.add_argument('--save-freq', type=int, default=100000)
parser.add_argument('--env-rgb', action='store_true', default=False)
parser.add_argument('--env-stacked', action='store_true', default=False)
parser.add_argument('--n-env', type=int, default=5)
parser.add_argument('--stack-frame', type=int, default=0)
parser.add_argument('--model-size', type=str, default='small')
parser.add_argument('--video-path', type=str, default='./tmp/tank1.avi')
parser.add_argument('--eval-game', type=int, default=50)
parser.add_argument('--save-path')
parser.add_argument('--checkpoint')
parser.add_argument('--policy_type', type=str, default='cnn')
parser.add_argument('--schedule', type=str, default='linear')
parser.add_argument('--death_penalty', action='store_true', default=False)
parser.add_argument('--friendly_fire', action='store_true', default=False)
parser.add_argument('--kill_bonus', action='store_true', default=False)
parser.add_argument('--eval-mode', action='store_true', default=False)
parser.add_argument('--testing', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--n-episode', type=int, default=5)
parser.add_argument('--record-stacked', action='store_true', default=False)
parser.add_argument('--record-rgb', action='store_true', default=False)
args = parser.parse_args()

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
#OBS_SPACES = [gym.spaces.box.Box(0,255,(128,128,4))]*NUM_LIVE_TANKS
OBS_SPACES = [gym.spaces.box.Box(0,255,(4,128,128))]*NUM_LIVE_TANKS
ACT_SPACES = [gym.spaces.box.Box(-1,1,(3,))]*NUM_LIVE_TANKS
