import math, time, random
import os
import sys
import argparse

import numpy as np

from make_env import make_env
from core.policy_record import PolicyRecord
from algos.torch_ppo.mappo_gpu_new import PPOPolicy as TorchGPUMAPPOPolicyNew
from algos.torch_ppo.callbacks import EvalCallback

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir',help='the location of saved policys and logs')
    parser.add_argument('--exe', help='the absolute path of the tanksworld executable')
    parser.add_argument('--teamname1', help='the name for team 1', default='red')
    parser.add_argument('--teamname2', help='the name for team 2', default='blue')
    parser.add_argument('--reward_weight', type=float, default=1.0)
    parser.add_argument('--penalty_weight', type=float, default=1.0)
    parser.add_argument('--ff_weight', type=float, default=0.0)
    parser.add_argument('--curriculum_start', type=float, default=-1)
    parser.add_argument('--curriculum_stop', type=float, default=-1)
    parser.add_argument('--policy_lr', type=float, default=3e-4)
    parser.add_argument('--value_lr', type=float, default=1e-3)
    parser.add_argument('--policy_lr_schedule', type=str, default='cons')
    parser.add_argument('--value_lr_schedule', type=str, default='cons')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--death_penalty', action='store_true', default=False)
    parser.add_argument('--friendly_fire', action='store_true', default=True)
    parser.add_argument('--kill_bonus', action='store_true', default=False)
    parser.add_argument('--eval_mode', action='store_true', default=False)
    parser.add_argument('--num_iter', type=int, default=1000)
    parser.add_argument('--env_seed', type=int, default=-1)
    parser.add_argument('--policy_seed', type=int, default=-1)
    parser.add_argument('--eval_checkpoint', type=str, default='')
    parser.add_argument('--save_tag', type=str, default='')
    parser.add_argument('--load_from_checkpoint', action='store_true', default=False)
    parser.add_argument('--seed_index', type=int, default=0)
    args = parser.parse_args()

    config = vars(args)
    folder_name = 'lrp={}{}__lrv={}{}__r={}__p={}__ff={}__H={}__{}'.format(config['policy_lr'],
                                                                         config['policy_lr_schedule'],
                                                                         config['value_lr'],
                                                                         config['value_lr_schedule'],
                                                                         config['reward_weight'],
                                                                         config['penalty_weight'],
                                                                         config['ff_weight'],
                                                                         config['batch_size'],
                                                                         args.save_tag)
    if config['curriculum_start'] >= 0.0:
        folder_name += '__CS={}__CF={}'.format(config['curriculum_start'], config['curriculum_stop'])
    folder_name += '/seed{}'.format(config['seed_index'])

    if isinstance(args.env_seed, int):
        env_seed = [args.env_seed]
    else:
        env_seed = args.env_seed

    stats_dir = './runs/stats_{}'.format(args.logdir)
    os.makedirs(stats_dir, exist_ok=True)

    env_kwargs = {'exe': args.exe,
                 'static_tanks': [], 'random_tanks': [5, 6, 7, 8, 9], 'disable_shooting': [],
                 'friendly_fire': True, 'kill_bonus': False, 'death_penalty': False,
                 'take_damage_penalty': True, 'tblogs': stats_dir,
                 'penalty_weight': config['penalty_weight'], 'reward_weight': 1.0,
                 'friendly_fire_weight': config['ff_weight'], 'timeout': 500, 'log_statistics': True,
                 'seed': args.env_seed, 'curriculum_stop': config['curriculum_stop'], 'curriculum_steps': args.num_iter}
    env = DummyVecEnv([lambda : make_env(**env_kwargs)])


    eval_env_kwargs = []
    _MAX_INT = 2147483647
    for _ in range(20):
        seed = np.random.randint(_MAX_INT)
        eval_env_kwargs.append({'exe': args.exe,
                  'static_tanks': [], 'random_tanks': [5, 6, 7, 8, 9], 'disable_shooting': [],
                  'friendly_fire': False, 'kill_bonus': False, 'death_penalty': False,
                  'take_damage_penalty': True, 'tblogs': stats_dir,
                  'penalty_weight': 1.0, 'reward_weight': 1.0,
                  'friendly_fire_weight': config['ff_weight'], 'timeout': 500, 'log_statistics': True,
                  'seed': seed, 'curriculum_stop': -1.0,
                  'curriculum_steps': args.num_iter})
    env_functions = []
    for i in range(2):
        env_functions.append(lambda : make_env(**eval_env_kwargs[i]))
    eval_env = SubprocVecEnv(env_functions)


    policy_record = PolicyRecord(folder_name, './logs/' + args.logdir + '/')
    model_id = '{}---{}-{}-{}'.format(args.logdir.split('/')[-1], folder_name, args.seed_index, args.save_tag)
    model_path = None
    if args.load_from_checkpoint:
        model_path = os.path.join(policy_record.data_dir, 'checkpoints', model_id)
        checkpoint_files = os.listdir(model_path)
        checkpoint_files.sort(key=lambda f: int(f.split('.')[0]))
        model_path = os.path.join(model_path, checkpoint_files[-1])

    policy_kwargs = {
        'steps_per_epoch': config['batch_size'],
        'train_pi_iters': 4,
        'train_v_iters': 4,
        'pi_lr': config['policy_lr'],
        'vf_lr': config['value_lr'],
        'ent_coef': -1.0,
        'pi_scheduler': config['policy_lr_schedule'],
        'vf_scheduler': config['value_lr_schedule'],
        'seed': args.policy_seed,
        'cnn_model_path': './models/frozen-cnn-0.8/4000000.pth',
        'model_path': model_path,
        'n_envs': 1,
        'model_id': model_id,
        'save_dir': os.path.join(policy_record.data_dir, 'checkpoints'),
        'freeze_rep': True,
    }

    callback = EvalCallback(env, policy_record, eval_env=eval_env)
    policy = TorchGPUMAPPOPolicyNew(env, callback, False, **policy_kwargs)
    policy.run(num_steps=args.num_iter)