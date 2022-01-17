# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
import os

# from tanksworld.make_env import make_env
import gym
import os
import argparse

parser = argparse.ArgumentParser(description="AI Safety TanksWorld")
parser.add_argument("--exe", help="the absolute path of the tanksworld executable")
parser.add_argument("--exp-dir", help="relative path for experiment")
parser.add_argument("--record", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--training", action="store_true", default=False)
parser.add_argument(
    "--continue-training", action="store_true", help="continute training"
)
parser.add_argument("--dummy-proc", action="store_true", default=False)
parser.add_argument("--n-steps", type=int, default=1024)
parser.add_argument("--warmup-steps", type=int, default=1000000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--config", type=int, default=1)
parser.add_argument("--n-tank-train", type=int, default=1)
parser.add_argument(
    "--extract-ftr-model", default="small", help="cnn model small or medium"
)
parser.add_argument(
    "--net-arch-size", type=int, default=64, help="net arch size for actor critic model"
)
parser.add_argument(
    "--n-input-enable",
    type=int,
    default=1,
    help="number of tanks input give for the algorithm",
)
parser.add_argument("--timestep", type=int, default=1000000)
parser.add_argument("--save-freq", type=int, default=10000)
parser.add_argument("--n-episode", type=int, default=5)
parser.add_argument("--n-envs", type=int, default=1)
parser.add_argument(
    "--env-timeout", type=int, default=500, help="Environments max timestep per episode"
)
parser.add_argument("--prune-threshold", type=float, default=0.1)
parser.add_argument("--shot-reward", action="store_true", default=False)
parser.add_argument("--shot-reward-amount", action="store_true", default=False)
parser.add_argument("--penalty-weight", type=float, default=1.0)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--ent-coef", type=float, default=0.00)
parser.add_argument(
    "--learning-rate-type", default="constant", help="constant or linear"
)
parser.add_argument("--input-type", default="stacked", help="stacked or dict")
parser.add_argument("--video-path", help="")
parser.add_argument("--save-path", help="")
parser.add_argument("--model-num", type=int, default=-1)
parser.add_argument("--load-type", help="none or full or cnn")
parser.add_argument("--freeze-cnn", action="store_true", default=False)
args = parser.parse_args()
params = vars(args)

# Tell the arena where it can put log files that describe the results of
# specific policies.  This is also used to pass results between root processes.

# LOG_COMMS_DIR = "logs/"+args.logdir+"/"
# os.makedirs(LOG_COMMS_DIR, exist_ok=True)

# Define where to find the environment
# if make_env is in this directory, os.getcwd() will suffice

MAKE_ENV_LOCATION = os.getcwd()


# Tell the arena what observation and action spaces to expect

# temp = make_env()

NUM_LIVE_TANKS = 5
OBS_SPACES = [gym.spaces.box.Box(0, 255, (128, 128, 4))] * NUM_LIVE_TANKS
ACT_SPACES = [gym.spaces.box.Box(-1, 1, (3,))] * NUM_LIVE_TANKS
