import os, subprocess
from subprocess import Popen
import json
import pdb


evaluated_params = []
exe = '/cis/home/kelgin/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64'
for main_folder in ['./logs/0.5']:
    for policy_folder in os.listdir(main_folder):
        if policy_folder[-4:] == 'json':
            continue

        if policy_folder.endswith('EVAL'):
            pi_lr = float(policy_folder.split('lrp=')[1].split('cons')[0])
            vf_lr = float(policy_folder.split('lrv=')[1].split('cons')[0])

            ckpt_directory = os.path.join(main_folder, policy_folder, '999999')
            if 'accumulated_stats.json' in os.listdir(ckpt_directory):
                evaluated_params.append((pi_lr, vf_lr))
            else:
                done = 0
                for seed_folder in os.listdir(ckpt_directory):
                    seed_directory = os.path.join(ckpt_directory, seed_folder)
                    with open(os.path.join(seed_directory, 'mean_statistics.json'), 'r') as f:
                        num_games = json.load(f)['Number of games']
                    if num_games == 100:
                        done += 1
                if done == 6:
                    evaluated_params.append((pi_lr, vf_lr))

for main_folder in ['./logs/0.5']:
    for policy_folder in os.listdir(main_folder):
        if policy_folder[-4:] == 'json':
            continue

        pi_lr = float(policy_folder.split('lrp=')[1].split('cons')[0])
        vf_lr = float(policy_folder.split('lrv=')[1].split('cons')[0])
        if (pi_lr, vf_lr) in evaluated_params:
            continue

        command = ['python3.6', 'grid_search_vectorized.py', '--exe', exe,
                   '--logdir', '0.5', '--n_env_seeds', '2',
                   '--n_policy_seeds', '3', '--ff_weight', '0.0', '--penalty_weight', '0.5',
                   '--policy_lr', '{}'.format(pi_lr), '--value_lr', '{}'.format(vf_lr), '--num_iter', '100',
                   '--batch_size', '64', '--eval_mode', '--eval_checkpoint', '999999',
                   '--eval_logdir', '0.5']
        print(' '.join(command))
        subprocess.call(command)