import json
import os
import pdb
from tensorboardX import SummaryWriter
import numpy as np
import gym

import trainer_config
from make_env import make_env
from core.policy_record import PolicyRecord
from algos.torch_ppo.mappo import PPOPolicy
from algos.torch_ppo.ippo import PPOPolicy as IPPOPolicy
#from algos.torch_ppo.vec_env import DummyVecEnv
#from algos.torch_ppo.vec_env import DummyVecEnv, SubprocVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from algos.torch_ppo.callbacks import EvalCallback


class Trainer:

    def __init__(self, config):
        self.config = config

    def get_folder_name(self):
        '''
        Sets up the folder name according to the configuration parameters
        :return:
        '''
        config = self.config

        folder_name = 'lrp={}__lrv={}__r={}__p={}__H={}'.format(config['policy_lr'],
                                                                config['value_lr'],
                                                                config['reward_weight'],
                                                                config['penalty_weight'],
                                                                config['batch_size'])
        if config['entropy_coef'] > 0.0: folder_name += '__ENT={}'.format(config['entropy_coef'])
        if config['beta']: folder_name += '__BETA'
        if config['curriculum_stop'] != -1: folder_name += '__CURR{}'.format(config['curriculum_stop'])
        if config['selfplay']: folder_name += '__SELFPLAY'
        if config['init_log_std'] != -0.5: folder_name += '__STD{}'.format(config['init_log_std'])
        if config['death_penalty']: folder_name += '__DP'
        if config['clip_ratio'] != 0.2: folder_name += '__CLIP{}'.format(config['clip_ratio'])
        if config['independent']: folder_name += '__IND'
        if config['local_std']: folder_name += '__LOCSTD'
        if config['num_workers'] > 1: folder_name += '__ROLLOUT={}'.format(config['num_workers'])
        if config['save_tag'] != '': folder_name += '__{}'.format(config['save_tag'])
        return folder_name



    def set_training_env(self):

        config = self.config
        folder_name = self.get_folder_name()

        all_training_envs = []
        all_policy_records = []
        all_policy_seeds = []
        all_tb_writers = []

        if config['num_workers'] == 1:

            # Set one policy per environment
            for e_seed_idx, e_seed in enumerate(self.env_seeds):
                for p_seed_idx, p_seed in enumerate(self.policy_seeds):
                    seed_idx = e_seed_idx * len(self.policy_seeds) + p_seed_idx

                    # This one is a dumb way of choosing one of the trained seeds
                    # Needs to be changed
                    if seed_idx == config['seed_idx']:

                        # Set folder name and policy record to plot the rewards
                        local_folder_name = folder_name + '/seed{}'.format(seed_idx)
                        if config['eval_mode']:
                            policy_record = PolicyRecord(local_folder_name, './logs/' + config['eval_logdir'] + '/',
                                                         std=True)
                        else:
                            policy_record = PolicyRecord(local_folder_name, './logs/' + config['logdir'] + '/',
                                                         std=True)

                        if not config['eval_mode']:
                            # Set TensorBoard writer
                            stats_dir = os.path.join('./logs', config['logdir'], local_folder_name, 'stats')
                            os.makedirs(stats_dir, exist_ok=True)
                            tb_writer = SummaryWriter(stats_dir)
                        else:
                            stats_dir = os.path.join('./junk')
                            os.makedirs(stats_dir, exist_ok=True)
                            tb_writer = SummaryWriter(stats_dir)

                        if config['selfplay'] or config['enemy_model'] is not None:
                            random_tanks = []
                            num_agents = 10
                        else:
                            random_tanks = [5, 6, 7, 8, 9]
                            num_agents = 5

                        env_kwargs = {'exe': config['exe'],
                                      'static_tanks': [], 'random_tanks': random_tanks, 'disable_shooting': [],
                                      'friendly_fire': False, 'kill_bonus': False,
                                      'death_penalty': config['death_penalty'],
                                      'take_damage_penalty': True, 'tblogs': stats_dir, 'tbwriter': tb_writer,
                                      'penalty_weight': config['penalty_weight'], 'reward_weight': 1.0,
                                      'timeout': 500, 'seed': e_seed,
                                      'curriculum_stop': config['curriculum_stop'],
                                      'use_state_vector': config['use_state_vector'],
                                      }

                        def make_env_(seed):
                            def init_():
                                env = make_env(**env_kwargs)
                                env._seed = seed
                                return env

                            return init_

                        def make_env_gym(seed):
                            def init_():
                                env = gym.make(self.config['env_name'])
                                return env

                            return init_

                        if not self.config['env_name'] == 'tanksworld':
                            make_env_ = make_env_gym
                            num_agents = 1

                        env = DummyVecEnv([make_env_(e_seed)], config['use_state_vector'], num_agents)
                        all_training_envs.append(env)
                        all_policy_records.append(policy_record)
                        all_policy_seeds.append(p_seed)
                        all_tb_writers.append(tb_writer)

        else: # Multiple rollouts

            # Set one policy per multiple environments
            for p_seed_idx, p_seed in enumerate(self.policy_seeds):
                seed_idx = p_seed_idx
                if seed_idx == config['seed_idx']:

                    # Set folder name and policy record to plot the rewards
                    local_folder_name = folder_name + '/seed{}'.format(seed_idx)
                    policy_record = PolicyRecord(local_folder_name, './logs/' + config['logdir'] + '/', std=True)

                    # Set TensorBoard writer
                    stats_dir = os.path.join('./logs', config['logdir'], local_folder_name, 'stats')
                    os.makedirs(stats_dir, exist_ok=True)
                    tb_writer = SummaryWriter(stats_dir)

                    random_tanks = [] if config['selfplay'] else [5,6,7,8,9]
                    env_kwargs = {'exe': config['exe'],
                                  'static_tanks': [], 'random_tanks': random_tanks, 'disable_shooting': [],
                                  'friendly_fire': False, 'kill_bonus': False, 'death_penalty': config['death_penalty'],
                                  'take_damage_penalty': True, 'tblogs': stats_dir,
                                  'penalty_weight': config['penalty_weight'], 'reward_weight': 1.0,
                                  'timeout': 500, 'curriculum_stop': config['curriculum_stop'],
                                  'use_state_vector': config['use_state_vector'],
                                  'attach_bounding_box': config['attach_bounding_box']
                                  }

                    def make_env_(seed):
                        def init_():
                            env = make_env(**env_kwargs)
                            env._seed = seed
                            return env

                        return init_

                    def make_env_gym(seed):
                        def init_():
                            env = gym.make(self.config['env_name'])
                            return env

                        return init_

                    if not self.config['env_name'] == 'tanksworld':
                        make_env_ = make_env_gym
                        num_agents = 1

                    env = SubprocVecEnv([make_env_(seed) for seed in self.env_seeds])
#                    env = DummyVecEnv([make_env_(seed) for seed in self.env_seeds])
                    

                    all_training_envs.append(env)
                    all_policy_records.append(policy_record)
                    all_policy_seeds.append(p_seed)
                    all_tb_writers.append(tb_writer)

        return all_training_envs, all_policy_records, all_policy_seeds, all_tb_writers



    def set_eval_env(self):

        config = self.config
        folder_name = self.get_folder_name()
        eval_folder_name = folder_name + '__EVAL'

        # Load or generate evaluation seeds
        eval_seed_folder = os.path.join('./logs', config['logdir'], 'eval_seeds.json')
        if os.path.exists(eval_seed_folder):
            with open(eval_seed_folder, 'r') as f:
                eval_env_seeds = json.load(f)
            eval_env_seeds = eval_env_seeds[:config['n_eval_seeds']]
        else:
            _MAX_INT = 2147483647  # Max int for Unity ML Seed
            eval_env_seeds = [np.random.randint(_MAX_INT) for _ in range(config['n_eval_seeds'])]
            with open(eval_seed_folder, 'w+') as f:
                json.dump(eval_env_seeds, f)

        all_eval_envs = []
        all_policy_records = []

        if config['num_workers'] == 1:  # One rollout
            for e_seed_idx, _ in enumerate(self.env_seeds):
                for p_seed_idx, _ in enumerate(self.policy_seeds):
                    seed_idx = e_seed_idx * len(self.policy_seeds) + p_seed_idx
                    if seed_idx == config['seed_idx']:
                        local_eval_folder_name = eval_folder_name + '/seed{}'.format(seed_idx)
                        policy_record = PolicyRecord(local_eval_folder_name, './logs/' + config['logdir'] + '/',
                                                     std=True)
                        stats_dir = './junk'

                        env_kwargs = {'exe': config['exe'],
                                      'static_tanks': [], 'random_tanks': [5, 6, 7, 8, 9], 'disable_shooting': [],
                                      'friendly_fire': False, 'kill_bonus': False, 'death_penalty': False,
                                      'take_damage_penalty': True, 'tblogs': stats_dir,
                                      'penalty_weight': config['penalty_weight'], 'reward_weight': 1.0,
                                      'timeout': 500}

                        def make_env_(seed):
                            def init_():
                                env = make_env(**env_kwargs)
                                env._seed = seed
                                return env

                            return init_

                        def make_env_gym(seed):
                            def init_():
                                env = gym.make(self.config['env_name'])
                                return env

                            return init_

                        if not self.config['env_name'] == 'tanksworld':
                            make_env_ = make_env_gym
                            num_agents = 1


                        
                        env = DummyVecEnv([make_env_(123)], config['use_state_vector'], num_agents)
                        all_eval_envs.append(env)
                        all_policy_records.append(policy_record)

        else: # Multiple rollouts

            for p_seed_idx, _ in enumerate(self.policy_seeds):
                seed_idx = p_seed_idx
                if seed_idx == config['seed_idx']:
                    local_eval_folder_name = eval_folder_name + '/seed{}'.format(seed_idx)
                    policy_record = PolicyRecord(local_eval_folder_name, './logs/' + config['logdir'] + '/', std=True)

                    env_kwargs = {'exe': config['exe'],
                                  'static_tanks': [], 'random_tanks': [5, 6, 7, 8, 9], 'disable_shooting': [],
                                  'friendly_fire': False, 'kill_bonus': False, 'death_penalty': False,
                                  'take_damage_penalty': True, 'tblogs': './junk',
                                  'penalty_weight': config['penalty_weight'], 'reward_weight': 1.0,
                                  'timeout': 500}

                    def make_env_(seed):
                        def init_():
                            env = make_env(**env_kwargs)
                            env._seed = seed
                            return env

                        return init_

                    env = SubprocVecEnv([make_env_(seed) for seed in self.env_seeds])
                    all_eval_envs.append(env)
                    all_policy_records.append(policy_record)

        return all_eval_envs, all_policy_records


    def get_policy_params(self):

        config = self.config

        policy_kwargs = {
            'batch_size': config['batch_size'],
            'rollout_length': config['rollout_length'],
            'train_pi_iters': config['optimzation_epochs'],
            'train_v_iters': config['optimzation_epochs'],
            'pi_lr': config['policy_lr'],
            'vf_lr': config['value_lr'],
            'clip_ratio': config['clip_ratio'],
            'cnn_model_path': config['cnn_path'] if config['cnn_path'] != 'None' else None,
            'n_envs': config['n_eval_seeds'] if config['eval_mode'] else config['num_workers'],
            'freeze_rep': config['freeze_rep'],
            'use_value_norm': config['valuenorm'],
            'use_beta': config['beta'],
            'init_log_std': config['init_log_std'],
            'selfplay': config['selfplay'],
            'centralized': config['centralized'],
            'centralized_critic': config['centralized_critic'],
            'use_state_vector': config['use_state_vector'],
            'local_std': config['local_std'],
            'enemy_model': config['enemy_model'],
            'env_name': config['env_name'],
            'single_agent': config['single_agent'],
            'hidden_sizes': config['hidden_sizes']
        }

        return policy_kwargs


    def set_seeds(self):

        config = self.config
        _MAX_INT = 2147483647  # Max int for Unity ML Seed
        n_policy_seeds = config['n_policy_seeds']
        n_rollout_threads = config['num_workers']
        n_env_seeds = n_rollout_threads if n_rollout_threads > 1 else config['n_env_seeds']

        # Create the log directory if not exists
        if not os.path.exists(os.path.join('./logs', config['logdir'])):
            os.mkdir(os.path.join('./logs', config['logdir']))

        # If seeds were saved under the log directory as a json file, load them.
        # Else generate new seeds and save.
        init_seeds = os.path.join('./logs', config['logdir'], 'seeds.json')

        if os.path.exists(init_seeds) and False:
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

    if args['eval_mode'] or args['visual_mode']:
        envs, eval_policy_records = trainer.set_eval_env()
#        _, train_policy_records, _, _ = trainer.set_training_env()

        model_path = 'logs/test/lrp=0.0003__lrv=0.001__r=1.0__p=0.3__H=64__ROLLOUT=7__lunar2/seed0/checkpoints'
        policies_to_run = []
        for seed_idx, policy_record in enumerate(eval_policy_records):

#            model_path = os.path.join(train_policy_records[seed_idx].data_dir, 'checkpoints')

            checkpoint_files = os.listdir(model_path)

            # If there is best checkpoint, load it
            # Else load the last checkpoint
            if 'best.pth' in checkpoint_files:
                model_path = os.path.join(model_path, 'best.pth')
            else:
                checkpoint_files.sort(key=lambda f: int(f.split('.')[0]))
                if len(checkpoint_files) > 0: model_path = os.path.join(model_path, checkpoint_files[-1])

            if not os.path.exists(model_path):
                pdb.set_trace()
            print('MODEL PATH', model_path)

            policy_params = trainer.get_policy_params()
            policy_params['model_path'] = model_path

            callback = None
            if args['env_name'] == 'tanksworld':
                callback = EvalCallback(envs[seed_idx], policy_record, eval_env=None)
            if args['independent']:
                policy = IPPOPolicy(envs[seed_idx], callback, True, **policy_params)
            else:
                policy = PPOPolicy(envs[seed_idx], callback, eval_mode=args['eval_mode'],
                                   visual_mode=args['visual_mode'], **policy_params)
            policies_to_run.append(policy)

    else:
        envs, policy_records, policy_seeds, tb_writers = trainer.set_training_env()

        policies_to_run = []
        for seed_idx, policy_record in enumerate(policy_records):

            model_path = None
            if args['load_from_checkpoint']:
                model_path = os.path.join(policy_record.data_dir, 'checkpoints')
                checkpoint_files = os.listdir(model_path)
                if 'best.pth' in checkpoint_files:
                    checkpoint_files.remove('best.pth')
                checkpoint_files.sort(key=lambda f: int(f.split('.')[0]))
                if len(checkpoint_files) > 0: model_path = os.path.join(model_path, checkpoint_files[-1])

            policy_params = trainer.get_policy_params()
            policy_params['model_path'] = model_path
            policy_params['tb_writer'] = tb_writers[seed_idx]
            policy_params['save_dir'] = os.path.join(policy_record.data_dir, 'checkpoints')
            policy_params['seed'] = policy_seeds[seed_idx]

            with open(os.path.join(policy_record.data_dir, 'parameters.json'), 'w+') as f:
                json.dump({'penalty_weight': args['penalty_weight'],
                           'batch_size': args['batch_size'],
                           'policy_lr': args['policy_lr'],
                           'value_lr': args['value_lr'],
                           'centralized': args['centralized'],
                           'freeze_rep': args['freeze_rep'],
                           'num_workers': args['num_workers']}, f, indent=4)

            # Set validation environment (with 3 random seeds)
            env_kwargs = {'exe': args['exe'],
                          'static_tanks': [], 'random_tanks': [5, 6, 7, 8, 9], 'disable_shooting': [],
                          'friendly_fire': False, 'kill_bonus': False, 'death_penalty': False,
                          'take_damage_penalty': True, 'tblogs': './junk',
                          'penalty_weight': args['penalty_weight'], 'reward_weight': 1.0,
                          'timeout': 500}


            def make_env_(seed):
                def init_():
                    env = make_env(**env_kwargs)
                    env._seed = seed
                    return env

                return init_


            _MAX_INT = 2147483647  # Max int for Unity ML Seed

            callback = None
            if args['env_name'] == 'tanksworld':
                val_seeds = [np.random.randint(_MAX_INT) for _ in range(3)]
#                val_env = SubprocVecEnv([make_env_(seed) for seed in val_seeds])
#                callback = EvalCallback(envs[seed_idx], policy_record, eval_env=val_env, eval_steps=100)
                callback = EvalCallback(envs[seed_idx], policy_record)#, eval_env=val_env, eval_steps=100)
            if args['independent']:
                policy = IPPOPolicy(envs[seed_idx], callback, False, **policy_params)
            else:
                policy = PPOPolicy(envs[seed_idx], callback, False, **policy_params)
            policies_to_run.append(policy)

    num_runs = args['num_iter'] if not args['eval_mode'] else args['num_eval_episodes']
    policies_to_run[0].run(num_steps=num_runs)
    envs[0].close()
    if args['eval_mode']: tb_writers[0].close()
