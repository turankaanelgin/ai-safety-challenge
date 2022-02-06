import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='the location of saved policys and logs')
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
parser.add_argument('--num_epochs', type=int, default=4)
parser.add_argument('--death_penalty', action='store_true', default=False)
parser.add_argument('--friendly_fire', action='store_true', default=True)
parser.add_argument('--kill_bonus', action='store_true', default=False)
parser.add_argument('--eval_mode', action='store_true', default=False)
parser.add_argument('--num_iter', type=int, default=1000)
parser.add_argument('--eval_checkpoint', type=int, default=999999)
parser.add_argument('--save_tag', type=str, default='')
parser.add_argument('--load_from_checkpoint', action='store_true', default=False)
parser.add_argument('--freeze_rep', action='store_true', default=False)
parser.add_argument('--use_rnn', action='store_true', default=False)
parser.add_argument('--eval_logdir', type=str, default='')
parser.add_argument('--multiplayer', action='store_true', default=False)
parser.add_argument('--valuenorm', action='store_true', default=False)
parser.add_argument('--beta', action='store_true', default=False)
parser.add_argument('--fixed_kl', action='store_true', default=False)
parser.add_argument('--adaptive_kl', action='store_true', default=False)
parser.add_argument('--kl_beta', type=float, default=3.0)
parser.add_argument('--num_rollout_threads', type=int, default=1)
parser.add_argument('--local_std', action='store_true', default=False)
parser.add_argument('--n_env_seeds', type=int, default=1)
parser.add_argument('--n_policy_seeds', type=int, default=1)
parser.add_argument('--cnn_path', type=str, default='./models/frozen-cnn-0.8/4000000.pth')
parser.add_argument('--weight_sharing', action='store_true', default=False)
parser.add_argument('--cuda_idx', type=int, default=0)

config = vars(parser.parse_args())

cuda_idx = config['cuda_idx']

command = []
for arg_name in config:
    if arg_name == 'cuda_idx': continue
    arg_value = config[arg_name]
    if isinstance(arg_value, bool):
        if arg_value:
            command += ['--{}'.format(arg_name)]
    else:
        if arg_value != '':
            command += ['--{}'.format(arg_name)]
            command += ['{}'.format(arg_value)]

for seed_idx in range(config['n_env_seeds']*config['n_policy_seeds']):
    command_to_run = 'CUDA_VISIBLE_DEVICES={} python3.6 trainer.py '.format(cuda_idx)
    command_to_run += ' '.join(command)
    command_to_run += ' --seed_idx {}'.format(seed_idx)

    with open('./tasks/task{}_cuda{}.sh'.format(seed_idx, cuda_idx), 'w+') as f:
        f.write('TMUX='' tmux new-session -s task{}_cuda{} '.format(seed_idx, cuda_idx))
        f.write('\'source ~/anaconda3/etc/profile.d/conda.sh\n')
        f.write('conda activate tanksworld\n')
        f.write(command_to_run)
        f.write('\'')
