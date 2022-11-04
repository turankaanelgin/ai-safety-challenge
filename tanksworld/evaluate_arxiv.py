import os
import json
from pprint import pprint
from collections import defaultdict


batch_sizes = []
pi_lr = []
vf_lr = []
penalty_weight = []
main_dir = '/scratch/telgin1/logs/tmp'
folders = []
not_processed = []
all_commands = []

'''
not_processed = defaultdict(list)
all_commands = []

for folder in os.listdir(main_dir):
    for seed_idx in range(6):
        if os.path.exists(os.path.join(main_dir, folder, 'seed{}'.format(seed_idx), 'mean_eval_statistics_v3.json')):
            with open(os.path.join(main_dir, folder, 'seed{}'.format(seed_idx), 'mean_eval_statistics_v3.json'),
                      'r') as f:
                stats = json.load(f)
            if stats['Number of games'] == 9 and stats['Number of environments'] == 30:
                print('DONE WITH FOLDER', folder)
                continue
            else:
                not_processed[folder].append(seed_idx)
        else:
            not_processed[folder].append(seed_idx)

pprint(not_processed)
with open('./to_train.json', 'w+') as f:
    json.dump(not_processed, f, indent=4)
exit(0)
'''

for seed_idx in range(6):
    commands = []
    for folder in os.listdir(main_dir):

        if os.path.exists(os.path.join(main_dir, folder, 'seed{}'.format(seed_idx), 'mean_eval_statistics_v3.json')):
            with open(os.path.join(main_dir, folder, 'seed{}'.format(seed_idx), 'mean_eval_statistics_v3.json'), 'r') as f:
                stats = json.load(f)
            if stats['Number of games'] == 9 and stats['Number of environments'] == 30:
                print('DONE WITH FOLDER', folder)
                continue
            else:
                not_processed.append(folder)
        elif not os.path.exists(os.path.join(main_dir, folder, 'seed{}'.format(seed_idx), 'checkpoints', '999999.pth')):
            continue
        else:
            not_processed.append(folder)

        folders.append(folder)
        try:
            param_pi = float(folder.split('lrp=')[1].split('__')[0])
            param_vf = float(folder.split('lrv=')[1].split('__')[0])
            param_batch = int(folder.split('H=')[1])
            penalty = float(folder.split('__p=')[1].split('__')[0])
        except:
            continue
        command_to_run = 'CUDA_VISIBLE_DEVICES=0 python trainer.py'
        command_to_run += ' --logdir tmp'
        command_to_run += ' --exe /home/telgin1/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64'
        command_to_run += ' --teamname1 red --teamname2 blue --reward_weight 1.0 --penalty_weight {}'.format(penalty)
        command_to_run += ' --ff_weight 0.0 --curriculum_stop -1 --policy_lr {} --value_lr {} --entropy_coef 0.0 --batch_size {}'.format(
            param_pi,
            param_vf,
            param_batch)
        command_to_run += ' --seed_idx {}'.format(seed_idx)
        command_to_run += ' --num_epochs 4 --clip_ratio 0.2 --friendly_fire --eval_mode --num_iter 100 --num_eval_episodes 10'
        command_to_run += ' --eval_logdir tmp --num_rollout_threads 1 --n_env_seeds 2 --n_policy_seeds 3'
        command_to_run += ' --cnn_path ./models/frozen-cnn-0.8/4000000.pth --init_log_std -0.5 --n_eval_seeds 30'
        command_to_run += '\npkill aisafetytanks_0'
        commands.append(command_to_run)
    all_commands += commands

    with open('./tasks/task_eval{}.sh'.format(seed_idx), 'w+') as f:
        for command_to_run in commands:
            f.write(command_to_run)
            f.write('\n')

    print(folders)

with open('./tasks/task_all_eval.sh', 'w+') as f:
    for command_to_run in all_commands:
        f.write(command_to_run)
        f.write('\n')