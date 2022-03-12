import os, subprocess
from subprocess import Popen
import json
import pdb
import pickle


for idx in range(6):
    with open('./tasks/commands{}_marcc.sh'.format(idx), 'w') as f:
        f.truncate()

exe = '/home/telgin1/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64'
'''
for main_eval_folder in ['./logs/final-baseline-eval']:
    for policy_folder in os.listdir(main_eval_folder):
        if policy_folder[-4:] == 'json':
            continue

        pi_lr = float(policy_folder.split('lrp=')[1].split('cons')[0])
        vf_lr = float(policy_folder.split('lrv=')[1].split('cons')[0])
        penalty = float(policy_folder.split('__p=')[1].split('__')[0])
        batch = int(policy_folder.split('__H=')[1].split('__')[0])
        evaluated_params.append((pi_lr, vf_lr, penalty, batch))
'''

seed_number = 3
with open('./evaluated_params_seed{}.pkl'.format(seed_number), 'rb') as f:
    evaluated_params = pickle.load(f)

for main_folder in ['./logs/final-baseline-v2-marcc']:
    for policy_folder in os.listdir(main_folder):
        if policy_folder.endswith('json'):
            continue

        pi_lr = float(policy_folder.split('lrp=')[1].split('cons')[0])
        vf_lr = float(policy_folder.split('lrv=')[1].split('cons')[0])
        penalty = float(policy_folder.split('__p=')[1].split('__')[0])
        batch = int(policy_folder.split('__H=')[1].split('__')[0])
        if (pi_lr, vf_lr, penalty, batch) in evaluated_params:
            continue

        command = ['python', 'grid_search_vectorized.py', '--exe', exe,
                   '--logdir', 'final-baseline-eval', '--eval_logdir', main_folder.split('/')[-1], '--n_env_seeds', '2',
                   '--n_policy_seeds', '3', '--ff_weight', '0.0', '--penalty_weight', '{}'.format(penalty),
                   '--policy_lr', '{}'.format(pi_lr), '--value_lr', '{}'.format(vf_lr), '--num_iter', '100',
                   '--batch_size', '{}'.format(batch), '--eval_mode', '--eval_checkpoint', '999999']
        print(' '.join(command))
        subprocess.call(command)
