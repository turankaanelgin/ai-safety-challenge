import os
import json
from collections import defaultdict


parameters = [
    #[0.0, 1e-3, 1e-3, 32],
    #[0.0, 1e-3, 5e-4, 32],
    #[0.0, 1e-3, 1e-4, 32],
    #[0.0, 3e-4, 1e-3, 32],
    [2.0, 3e-4, 5e-4, 128],
    #[0.5, 1e-3, 1e-3, 128],
    #[0.5, 1e-3, 5e-4, 128],
    #[0.5, 3e-4, 5e-4, 128]
    #[0.0, 3e-4, 1e-4, 32]
]

main_dir = 'leftover-baseline'
all_commands = []

for seed_idx in range(6):
    commands = []

    for param_list in parameters:
        penalty = param_list[0]
        param_pi = param_list[1]
        param_vf = param_list[2]
        param_batch = param_list[3]

        command_to_run = 'CUDA_VISIBLE_DEVICES=0 python trainer.py'
        command_to_run += ' --logdir {}'.format(main_dir)
        command_to_run += ' --exe /home/telgin1/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64'
        command_to_run += ' --teamname1 red --teamname2 blue --reward_weight 1.0 --penalty_weight {}'.format(penalty)
        command_to_run += ' --ff_weight 0.0 --curriculum_stop -1 --policy_lr {} --value_lr {} --entropy_coef 0.0 --batch_size {}'.format(
            param_pi,
            param_vf,
            param_batch)
        command_to_run += ' --seed_idx {}'.format(seed_idx)
        command_to_run += ' --num_epochs 4 --clip_ratio 0.2 --friendly_fire --num_iter 1000000 --num_eval_episodes 10'
        command_to_run += ' --eval_logdir {} --num_rollout_threads 1 --n_env_seeds 2 --n_policy_seeds 3'.format(main_dir)
        command_to_run += ' --cnn_path /scratch/telgin1/models/frozen-cnn-0.8/4000000.pth --init_log_std -0.5 --n_eval_seeds 30 --freeze_rep'
        commands.append(command_to_run)

    all_commands += commands

    with open('./tasks/task{}.sh'.format(seed_idx), 'w+') as f:
        for command_to_run in commands:
            f.write(command_to_run)
            f.write('\n')

with open('./tasks/task_all.sh', 'w+') as f:
    for command_to_run in all_commands:
        f.write(command_to_run)
        f.write('\n')
