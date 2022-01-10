import json
import math, time, random
import os
import pdb
import sys
import argparse
from tensorboardX import SummaryWriter

import numpy as np

from make_env import make_env
from core.policy_record import PolicyRecord
from algos.torch_ppo.mappo_gpu_new import PPOPolicy as TorchGPUMAPPOPolicyNew
from algos.torch_ppo.mappo_gpu_multiplayer import PPOPolicy as MAPPOMultiPlayer
from algos.torch_ppo.mappo_gpu_curiosity import PPOPolicy as MAPPOCuriosity
from algos.torch_sac.sac_new import SACPolicy
from algos.maddpg.maddpg import MADDPGPolicy
from algos.torch_ppo.callbacks import EvalCallback
from algos.torch_ppo.vec_env import DummyVecEnv, SubprocVecEnv


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
    parser.add_argument('--entropy_coef', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--death_penalty', action='store_true', default=False)
    parser.add_argument('--friendly_fire', action='store_true', default=True)
    parser.add_argument('--kill_bonus', action='store_true', default=False)
    parser.add_argument('--eval_mode', action='store_true', default=False)
    parser.add_argument('--num_iter', type=int, default=1000)
    parser.add_argument('--env_seed', type=int, default=-1)
    parser.add_argument('--policy_seed', type=int, default=-1)
    parser.add_argument('--eval_checkpoint', type=int, default=999999)
    parser.add_argument('--save_tag', type=str, default='')
    parser.add_argument('--load_from_checkpoint', action='store_true', default=False)
    parser.add_argument('--seed_index', type=int, default=0)
    parser.add_argument('--freeze_rep', action='store_true', default=False)
    parser.add_argument('--use_rnn', action='store_true', default=False)
    parser.add_argument('--eval_logdir', type=str, default='')
    parser.add_argument('--multiplayer', action='store_true', default=False)
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
    if config['entropy_coef'] > 0.0:
        folder_name += '__ENT={}'.format(config['entropy_coef'])
    if config['multiplayer']:
        folder_name += '__MULTI'
    eval_folder_name = folder_name
    if args.eval_mode:
        eval_folder_name += '__EVAL/{}'.format(args.eval_checkpoint)
    folder_name += '/seed{}'.format(config['seed_index'])
    eval_folder_name += '/seed{}'.format(config['seed_index'])

    if isinstance(args.env_seed, int):
        env_seed = [args.env_seed]
    else:
        env_seed = args.env_seed

    stats_dir = os.path.join('./logs', args.logdir, folder_name, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    tb_writer = SummaryWriter(stats_dir)

    if args.eval_mode:
        eval_seed_folder = os.path.join('./logs', args.logdir, 'eval_seeds.json')
        if os.path.exists(eval_seed_folder):
            with open(eval_seed_folder, 'r') as f:
                env_seeds = json.load(f)
        else:
            _MAX_INT = 2147483647  # Max int for Unity ML Seed
            env_seeds = [np.random.randint(_MAX_INT) for _ in range(10)]
            with open(eval_seed_folder, 'w+') as f:
                json.dump(env_seeds, f)

        env_kwargs = []
        for idx in range(10):
            env_kwargs.append({'exe': args.exe,
                               'static_tanks': [], 'random_tanks': [5,6,7,8,9], 'disable_shooting': [],
                               'friendly_fire': True, 'kill_bonus': False, 'death_penalty': False,
                               'take_damage_penalty': True, 'tblogs': stats_dir,
                               'penalty_weight': config['penalty_weight'], 'reward_weight': 1.0,
                               'friendly_fire_weight': config['ff_weight'], 'timeout': 500, 'log_statistics': True,
                               'seed': env_seeds[idx], 'curriculum_stop': config['curriculum_stop'], 'curriculum_steps': args.num_iter})
        env_functions = []
        for idx in range(len(env_kwargs)):
            env_functions.append(lambda : make_env(**env_kwargs[idx]))
        stacked_env = [env_functions[i] for i in range(len(env_kwargs))]
        env = SubprocVecEnv(stacked_env)

    else:
        if args.multiplayer:
            random_tanks = []
            num_agents = 10
        else:
            random_tanks = [5,6,7,8,9]
            num_agents = 5
        env_kwargs = {'exe': args.exe,
                     'static_tanks': [], 'random_tanks': random_tanks, 'disable_shooting': [],
                     'friendly_fire': True, 'kill_bonus': False, 'death_penalty': False,
                     'take_damage_penalty': True, 'tblogs': stats_dir, 'tb_writer': tb_writer,
                     'penalty_weight': config['penalty_weight'], 'reward_weight': 1.0,
                     'friendly_fire_weight': config['ff_weight'], 'timeout': 500, 'log_statistics': True,
                     'seed': args.env_seed, 'curriculum_stop': config['curriculum_stop'], 'curriculum_steps': args.num_iter,}
        env = DummyVecEnv([lambda : make_env(**env_kwargs)], num_agents)

    if args.eval_mode:
        policy_record = PolicyRecord(eval_folder_name, './logs/' + args.logdir + '/')
    else:
        policy_record = PolicyRecord(folder_name, './logs/' + args.logdir + '/')
    model_id = '{}---{}-{}-{}'.format('final-baseline-v2', folder_name, args.seed_index, args.save_tag)
    model_path = None
    if args.load_from_checkpoint:
        model_path = os.path.join(policy_record.data_dir, 'checkpoints', model_id)
        checkpoint_files = os.listdir(model_path)
        checkpoint_files.sort(key=lambda f: int(f.split('.')[0]))
        model_path = os.path.join(model_path, checkpoint_files[-1])
    elif args.eval_mode:
        model_path = os.path.join('./logs', args.eval_logdir, folder_name, 'checkpoints', model_id)
        model_path = os.path.join(model_path, '{}.pth'.format(args.eval_checkpoint))

    policy_kwargs = {
        'steps_per_epoch': config['batch_size'],
        'train_pi_iters': 4,
        'train_v_iters': 4,
        'pi_lr': config['policy_lr'],
        'vf_lr': config['value_lr'],
        'ent_coef': config['entropy_coef'],
        'pi_scheduler': config['policy_lr_schedule'],
        'vf_scheduler': config['value_lr_schedule'],
        'seed': args.policy_seed,
        'cnn_model_path': './models/frozen-cnn-0.8/4000000.pth',
        'enemy_model_path': '/cis/net/r09_ssd/data/kelgin/final-baseline-v2-bernese/'+\
                            'lrp=0.0003cons__lrv=0.001cons__r=1.0__p=0.5__ff=0.0__H=64__/seed0/checkpoints/'+
                            'final-baseline-v2---lrp=0.0003cons__lrv=0.001cons__r=1.0__p=0.5__ff=0.0__H=64__/'+
                            'seed0-0-/999999.pth',
        'model_path': model_path,
        'n_envs': 1,
        'model_id': model_id,
        'save_dir': os.path.join(policy_record.data_dir, 'checkpoints'),
        'freeze_rep': args.freeze_rep,
        'use_rnn': args.use_rnn,
        'num_states': 5,
        'tb_writer': tb_writer,
    }

    callback = EvalCallback(env, policy_record, eval_env=None)
    policy = TorchGPUMAPPOPolicyNew(env, callback, args.eval_mode, **policy_kwargs)
    #policy = MAPPOMultiPlayer(env, callback, args.eval_mode, **policy_kwargs)
    policy.run(num_steps=args.num_iter)

    '''
    policy_kwargs = {
        'steps_per_epoch': config['batch_size'],
        'seed': args.policy_seed,
        'model_path': None,
        'cnn_model_path': './models/frozen-cnn-0.8/4000000.pth',
        'freeze_rep': args.freeze_rep,
        'save_dir': os.path.join(policy_record.data_dir, 'checkpoints'),
        'model_id': model_id,
        'n_envs': 1,
    }
    
    callback = EvalCallback(env, policy_record, eval_env=None)
    #policy = SACPolicy(env, callback, args.eval_mode, **policy_kwargs)
    policy = MAPPOCuriosity(env, callback, args.eval_mode, **policy_kwargs)
    #policy = MADDPGPolicy(env, callback, args.eval_mode, **policy_kwargs)
    policy.run(num_steps=args.num_iter)
    '''