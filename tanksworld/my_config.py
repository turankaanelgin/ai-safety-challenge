# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import os
#from tanksworld.make_env import make_env
import gym
import os
import argparse

parser = argparse.ArgumentParser(description='AI Safety TanksWorld')
#parser.add_argument('--logdir',help='the location of saved policys and logs')
parser.add_argument('--exe', help='the absolute path of the tanksworld executable')
#parser.add_argument('--teamname1', help='the name for team 1')
#parser.add_argument('--teamname2', help='the name for team 2')
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--training', action='store_true', default=False)
parser.add_argument('--dummy-proc', action='store_true', default=False)
parser.add_argument('--n-steps', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--timestep', type=int, default=1000000)
parser.add_argument('--save-freq', type=int, default=10000)
parser.add_argument('--n-episode', type=int, default=5)
parser.add_argument('--n-envs', type=int, default=1)
parser.add_argument('--penalty-weight', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--ent-coef', type=float, default=0.00)
parser.add_argument('--lr-type', default='constant', help='constant or linear')
parser.add_argument('--video-path', help='the absolute path of the tanksworld executable')
parser.add_argument('--save-path', help='the absolute path of the tanksworld executable')
args = parser.parse_args()

# Tell the arena where it can put log files that describe the results of
# specific policies.  This is also used to pass results between root processes.

#LOG_COMMS_DIR = "logs/"+args.logdir+"/"
#os.makedirs(LOG_COMMS_DIR, exist_ok=True)

# Define where to find the environment
# if make_env is in this directory, os.getcwd() will suffice

MAKE_ENV_LOCATION = os.getcwd()


# Tell the arena what observation and action spaces to expect

#temp = make_env()

NUM_LIVE_TANKS = 5
OBS_SPACES = [gym.spaces.box.Box(0,255,(128,128,4))]*NUM_LIVE_TANKS
ACT_SPACES = [gym.spaces.box.Box(-1,1,(3,))]*NUM_LIVE_TANKS
