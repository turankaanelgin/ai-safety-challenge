import os
import subprocess
import argparse
import json
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
seed = parser.parse_args().seed

main_folder = './logs/final-baseline-v2-portable'

evaluated = []
main_eval_folder = './logs/final-baseline-eval-final'
for policy_folder in os.listdir(main_eval_folder):

    if policy_folder.endswith('json'):
        continue

    pi_lr = float(policy_folder.split('lrp=')[1].split('__')[0])
    vf_lr = float(policy_folder.split('lrv=')[1].split('__')[0])
    penalty = float(policy_folder.split('__p=')[1].split('__')[0])
    batch = int(policy_folder.split('__H=')[1].split('__')[0])

    for seed_idx, seed_folder in enumerate(os.listdir(os.path.join(main_eval_folder, policy_folder, '999999'))):
        if seed_idx == seed:
            stats_file = os.path.join(main_eval_folder, policy_folder, '999999', seed_folder, 'mean_statistics.json')
            if not os.path.exists(stats_file): continue

            with open(stats_file, 'r') as f:
                stats = json.load(f)

            if stats['Number of games'] == 100:
                evaluated.append((pi_lr, vf_lr, penalty, batch))

print('EVALUATED', evaluated)
with open(os.path.join(main_folder, 'evaluated{}'.format(seed)), 'wb') as f:
    pickle.dump(evaluated, f)

for policy_folder in os.listdir(main_folder):

    if policy_folder.endswith('json') or 'evaluated' in policy_folder:
        continue

    pi_lr = float(policy_folder.split('lrp=')[1].split('__')[0])
    vf_lr = float(policy_folder.split('lrv=')[1].split('__')[0])
    penalty = float(policy_folder.split('__p=')[1].split('__')[0])
    batch = int(policy_folder.split('__H=')[1].split('__')[0])

    if (pi_lr, vf_lr, penalty, batch) in evaluated:
        print('SKIPPING', (pi_lr, vf_lr, penalty, batch))
        continue

    command = ['python', 'trainer.py', '--exe',
               '/home/telgin1/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64',
               '--reward_weight', '1.0', '--penalty_weight', str(penalty), '--batch_size', str(batch),
               '--policy_lr', str(pi_lr), '--value_lr', str(vf_lr),
               '--eval_mode', '--logdir', 'final-baseline-eval-final',
               '--eval_logdir', 'final-baseline-v2-portable', '--num_iter', '100',
               '--seed_idx', str(seed), '--n_env_seeds', '2', '--n_policy_seeds', '3']
    print(' '.join(command))
    subprocess.call(command)

    evaluated.append((pi_lr, vf_lr, penalty, batch))
