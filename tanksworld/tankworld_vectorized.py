# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
# To create a script for the arena, we need to interact with a UserStem object.
# This object is active only on the root process, and broadcasts commands to the
# other processes.

import math, time, random
import os
import numpy as np

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *
from arena5.core.env_process import EnvironmentProcess 
import argparse
import json
import shutil

from algos.improved_ppo.ppo import PPOPolicy as ImprovedPPOPolicy
from algos.torch_ppo.ppo import PPOPolicy 
import my_config as cfg
from env import TanksWorldEnv
from mpi4py import MPI

comm_world = MPI.COMM_WORLD

args = cfg.args

additional_policies = {'improved_ppo': PPOPolicy}
#arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES, additional_policies)

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
def make_env(**kwargs):
    return TanksWorldEnv(cfg.args.exe, **kwargs)
match_group_comm = comm_world
root_proc = 0
will_call_render = 0
rank = comm_world.Get_rank()
size = comm_world.Get_size()
n_env = size - 1
num_tank = 5
class TanksworldVectorizedEnv():
    def __init__(self, comm, root_proc):
        self.root_proc = root_proc

    def step(self, actions):
        actions = actions.tolist()
        send_actions = [None]
        for i in range(size - 1):
            env_actions = [[[-1],[0]]]
            for index, act in enumerate(actions[i * num_tank: (i+1) * num_tank]):
                env_actions.append([[index], [act]])
            send_actions.append(env_actions)
        comm_world.scatter(send_actions, root=self.root_proc)
        gather_state = comm_world.gather(None, root=self.root_proc)
        obs, rewards, dones = [], [], []
 
        for env_idx in range(1, size):
            for o in gather_state[env_idx][0]:
                obs.append(o)
            rewards += gather_state[env_idx][1]
            dones += [gather_state[env_idx][2]] * num_tank
        #mpi_print(76, len(obs), obs[0].shape, rewards, dones)
        return obs, rewards, dones, {}


if rank == 0:
    state = None
    state = comm_world.gather(state, root=root_proc)
    env = TanksworldVectorizedEnv(comm_world, 0)
    while True:
        actions = np.random.rand((size - 1) * 5, 3)
        env.step(actions)

else:
    env = EnvironmentProcess(make_env, comm_world, match_group_comm, root_proc, will_call_render, env_kwargs=kwargs_1)
    env.proxy_sync() 
    env.run(10000)
