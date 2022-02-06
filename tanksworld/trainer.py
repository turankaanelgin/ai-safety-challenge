import json
import math, time, random
import os
import pdb
import sys
import argparse
from tensorboardX import SummaryWriter
import numpy as np
from multiprocessing import Process

import trainer_config
from make_env import make_env
from core.policy_record import PolicyRecord
#from algos.torch_ppo.mappo_gpu_new import PPOPolicy as TorchGPUMAPPOPolicyNew
from algos.torch_ppo.mappo_gpu_new_improved import PPOPolicy as TorchGPUMAPPOPolicyNew
from algos.torch_ppo.vec_env import DummyVecEnv, SubprocVecEnv
from algos.torch_ppo.callbacks import EvalCallback


class Trainer:
    def __init__(self, config):
        self.config = config

    def get_folder_name(self):
        config = self.config
        assert not (config['fixed_kl'] and config['adaptive_kl']), 'Fixed and adaptive KL cannot both be True'

        folder_name = 'lrp={}__lrv={}__r={}__p={}__H={}'.format(config['policy_lr'],
                                                                    config['value_lr'],
                                                                    config['reward_weight'],
                                                                    config['penalty_weight'],
                                                                    config['batch_size'])
        if config['entropy_coef'] > 0.0: folder_name += '__ENT={}'.format(config['entropy_coef'])
        if config['beta']: folder_name += '__BETA'
        if config['fixed_kl']:
            folder_name += '__FIXEDKL{}'.format(config['kl_beta'])
        elif config['adaptive_kl']:
            folder_name += '__ADAPTIVEKL{}'.format(config['kl_beta'])
        if config['use_rnn']: folder_name += '__RNN'
        if config['local_std']: folder_name += '__LOCALSTD'
        if config['num_rollout_threads'] > 1: folder_name += '__ROLLOUT={}'.format(config['num_rollout_threads'])
        if config['save_tag'] != '': folder_name += '__{}'.format(config['save_tag'])
        return folder_name

    def set_training_env(self):

        config = self.config
        folder_name = self.get_folder_name()

        all_training_envs = []
        all_policy_records = []
        all_policy_seeds = []
        all_tb_writers = []

        if config['num_rollout_threads'] == 1: # One rollout

            # Set one policy per environment
            for e_seed_idx, e_seed in enumerate(self.env_seeds):
                for p_seed_idx, p_seed in enumerate(self.policy_seeds):
                    seed_idx = e_seed_idx*len(self.policy_seeds)+p_seed_idx

                    # Set folder name and policy record to plot the rewards
                    local_folder_name = folder_name + '/seed{}'.format(seed_idx)
                    policy_record = PolicyRecord(local_folder_name, './logs/' + config['logdir'] + '/')

                    # Set TensorBoard writer
                    stats_dir = os.path.join('./logs', config['logdir'], local_folder_name, 'stats')
                    os.makedirs(stats_dir, exist_ok=True)
                    tb_writer = SummaryWriter(stats_dir)

                    env_kwargs = {'exe': config['exe'],
                                  'static_tanks': [], 'random_tanks': [5,6,7,8,9], 'disable_shooting': [],
                                  'friendly_fire': False, 'kill_bonus': False, 'death_penalty': False,
                                  'take_damage_penalty': True, 'tblogs': stats_dir, 'tbwriter': tb_writer,
                                  'penalty_weight': config['penalty_weight'], 'reward_weight': 1.0,
                                  'timeout': 500, 'seed': e_seed, }
                    env = DummyVecEnv([lambda: make_env(**env_kwargs)], 5)
                    all_training_envs.append(env)
                    all_policy_records.append(policy_record)
                    all_policy_seeds.append(p_seed)
                    all_tb_writers.append(tb_writer)
        else: # Multiple rollouts

            # Set one policy per multiple environments
            for p_seed_idx, p_seed in enumerate(self.policy_seeds):
                seed_idx = p_seed_idx

                # Set folder name and policy record to plot the rewards
                local_folder_name = folder_name + '/seed{}'.format(seed_idx)
                policy_record = PolicyRecord(local_folder_name, './logs/' + config['logdir'] + '/')

                # Set TensorBoard writer
                stats_dir = os.path.join('./logs', config['logdir'], local_folder_name, 'stats')
                os.makedirs(stats_dir, exist_ok=True)
                tb_writer = SummaryWriter(stats_dir)

                env_kwargs = []
                for e_seed in self.env_seeds:
                    env_kwargs.append({'exe': config['exe'],
                                       'static_tanks': [], 'random_tanks': [5, 6, 7, 8, 9], 'disable_shooting': [],
                                       'friendly_fire': False, 'kill_bonus': False, 'death_penalty': False,
                                       'take_damage_penalty': True, 'tblogs': stats_dir,
                                       'penalty_weight': config['penalty_weight'], 'reward_weight': 1.0,
                                       'timeout': 500, 'seed': e_seed})
                env_functions = []
                for idx in range(len(env_kwargs)):
                    env_functions.append(lambda: make_env(**env_kwargs[idx]))
                env = SubprocVecEnv(env_functions)
                all_training_envs.append(env)
                all_policy_records.append(policy_record)
                all_policy_seeds.append(p_seed)
                all_tb_writers.append(tb_writer)

        return all_training_envs, all_policy_records, all_policy_seeds, all_tb_writers


    def set_eval_env(self):

        def construct_env():
            env_kwargs = []
            for e_seed in eval_env_seeds:
                env_kwargs.append({'exe': config['exe'],
                                   'static_tanks': [], 'random_tanks': [5,6,7,8,9], 'disable_shooting': [],
                                   'friendly_fire': False, 'kill_bonus': False, 'death_penalty': False,
                                   'take_damage_penalty': True, 'tblogs': stats_dir, 'tbwriter': tb_writer,
                                   'penalty_weight': config['penalty_weight'], 'reward_weight': 1.0,
                                   'timeout': 500, 'seed': e_seed, })

            env_functions = []
            for idx in range(len(env_kwargs)):
                env_functions.append(lambda: make_env(**env_kwargs[idx]))
            env = SubprocVecEnv(env_functions)
            return env

        config = self.config
        folder_name = self.get_folder_name()
        eval_folder_name = folder_name + '__EVAL/{}'.format(config['eval_checkpoint'])

        # Load or generate evaluation seeds
        eval_seed_folder = os.path.join('./logs', config['logdir'], 'eval_seeds.json')
        if os.path.exists(eval_seed_folder):
            with open(eval_seed_folder, 'r') as f:
                eval_env_seeds = json.load(f)
        else:
            _MAX_INT = 2147483647  # Max int for Unity ML Seed
            eval_env_seeds = [np.random.randint(_MAX_INT) for _ in range(10)]
            with open(eval_seed_folder, 'w+') as f:
                json.dump(eval_env_seeds, f)

        all_eval_envs = []
        all_policy_records = []

        if config['num_rollout_threads'] == 1:  # One rollout

            for e_seed_idx, _ in enumerate(self.env_seeds):
                for p_seed_idx, _ in enumerate(self.policy_seeds):
                    seed_idx = e_seed_idx * len(self.policy_seeds) + p_seed_idx
                    local_eval_folder_name = eval_folder_name + '/seed{}'.format(seed_idx)
                    policy_record = PolicyRecord(local_eval_folder_name, './logs/' + config['logdir'] + '/')
                    env = construct_env()
                    all_eval_envs.append(env)
                    all_policy_records.append(policy_record)

        else: # Multiple rollouts

            for p_seed_idx, _ in enumerate(self.policy_seeds):
                seed_idx = p_seed_idx
                local_eval_folder_name = eval_folder_name + '/seed{}'.format(seed_idx)
                policy_record = PolicyRecord(local_eval_folder_name, './logs' + config['logdir'] + '/')
                env = construct_env()
                all_eval_envs.append(env)
                all_policy_records.append(policy_record)

        return all_eval_envs, all_policy_records


    def get_policy_params(self):
        config = self.config

        policy_kwargs = {
            'steps_per_epoch': config['batch_size'],
            'train_pi_iters': config['num_epochs'],
            'train_v_iters': config['num_epochs'],
            'pi_lr': config['policy_lr'],
            'vf_lr': config['value_lr'],
            'entropy_coef': config['entropy_coef'],
            'pi_scheduler': config['policy_lr_schedule'],
            'vf_scheduler': config['value_lr_schedule'],
            'cnn_model_path': config['cnn_path'] if config['cnn_path'] != 'None' else None,
            'n_envs': config['num_rollout_threads'],
            'freeze_rep': config['freeze_rep'],
            'use_rnn': config['use_rnn'],
            'num_states': 4,
            'use_value_norm': config['valuenorm'],
            'use_beta': config['beta'],
            'use_fixed_kl': config['fixed_kl'],
            'use_adaptive_kl': config['adaptive_kl'],
            'kl_beta': config['kl_beta'],
            'local_std': config['local_std'],
            'weight_sharing': config['weight_sharing'],
        }

        return policy_kwargs


    def set_seeds(self):
        config = self.config
        _MAX_INT = 2147483647  # Max int for Unity ML Seed
        n_policy_seeds = config['n_policy_seeds']
        n_rollout_threads = config['num_rollout_threads']
        n_env_seeds = n_rollout_threads if n_rollout_threads > 1 else config['n_env_seeds']

        # Create the log directory if not exists
        if not os.path.exists(os.path.join('./logs', config['logdir'])):
            os.mkdir(os.path.join('./logs', config['logdir']))

        # If seeds were saved under the log directory as a json file, load them.
        # Else generate new seeds and save.
        init_seeds = os.path.join('./logs', config['logdir'], 'seeds.json')

        if os.path.exists(init_seeds):
            with open(init_seeds, 'r') as f:
                seeds = json.load(f)
                env_seeds = seeds['env_seeds']
                policy_seeds = seeds['policy_seeds']

                # If there are more seeds than needed, take the first ones
                if n_env_seeds < len(env_seeds):
                    env_seeds = env_seeds[:n_env_seeds]
                if n_policy_seeds < len(policy_seeds):
                    policy_seeds = policy_seeds[:n_policy_seeds]
        else:
            env_seeds = [np.random.randint(_MAX_INT) for _ in range(n_env_seeds)]
            policy_seeds = [np.random.randint(_MAX_INT) for _ in range(n_policy_seeds)]
            with open(init_seeds, 'w+') as f:
                json.dump({'env_seeds': env_seeds,
                           'policy_seeds': policy_seeds}, f)

        self.env_seeds = env_seeds
        self.policy_seeds = policy_seeds



if __name__=='__main__':

    def run_policy(pol, env):
        pol.run(num_steps=args['num_iter'])
        env.close()

    args = trainer_config.config
    trainer = Trainer(args)
    trainer.set_seeds()

    if args['eval_mode']:
        envs, policy_records = trainer.set_eval_env()

        policies_to_run = []
        for seed_idx, policy_record in enumerate(policy_records):

            model_path = os.path.join('./logs', args['eval_logdir'], policy_record.data_dir, 'checkpoints')
            model_path = os.path.join(model_path, '{}.pth'.format(args['eval_checkpoint']))
            if not os.path.exists(model_path):
                print('CHECKPOINT DOES NOT EXIST')
                exit(0)

            policy_params = trainer.get_policy_params()
            policy_params['model_path'] = model_path
            policy_params['save_dir'] = os.path.join(policy_record.data_dir, 'checkpoints')
            policy_params['tb_writer'] = tbwriter

            callback = EvalCallback(envs[seed_idx], policy_record, eval_env=None)
            policy = TorchGPUMAPPOPolicyNew(envs[seed_idx], callback, True, **policy_params)
            policies_to_run.append(policy)

    else:
        envs, policy_records, policy_seeds, tb_writers = trainer.set_training_env()

        policies_to_run = []
        for seed_idx, policy_record in enumerate(policy_records):

            model_path = None
            if args['load_from_checkpoint']:
                model_path = os.path.join(policy_record.data_dir, 'checkpoints')
                checkpoint_files = os.listdir(model_path)
                checkpoint_files.sort(key=lambda f: int(f.split('.')[0]))
                if len(checkpoint_files) > 0: model_path = os.path.join(model_path, checkpoint_files[-1])

            policy_params = trainer.get_policy_params()
            policy_params['model_path'] = model_path
            policy_params['tb_writer'] = tb_writers[seed_idx]
            policy_params['save_dir'] = os.path.join(policy_record.data_dir, 'checkpoints')
            policy_params['seed'] = policy_seeds[seed_idx]

            callback = EvalCallback(envs[seed_idx], policy_record, eval_env=envs[seed_idx], eval_steps=5)
            policy = TorchGPUMAPPOPolicyNew(envs[seed_idx], callback, False, **policy_params)
            policies_to_run.append(policy)


    if len(policies_to_run) == 1:
        policies_to_run[0].run(num_steps=args['num_iter'])
        envs[0].close()
    else:
        # TODO make this more clever
        seed_idx_to_run = args['seed_idx']
        policies_to_run[seed_idx_to_run].run(num_steps=args['num_iter'])
        envs[seed_idx_to_run].close()
        '''
        processes = []
        for env, policy in zip(envs, policies_to_run):
            proc = Process(target=run_policy, args=(policy, env))
            proc.start()
            processes.append(proc)
        for proc in processes:
            proc.join()
        '''