#!/bin/bash
# Basic while loop
CUDA_VISIBLE_DEVICES=0 python3.6 grid_search_vectorized.py --exe ~/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --logdir final-baseline-eval --n_env_seeds 2 --n_policy_seeds 3 --ff_weight 1.0 --penalty_weight 0.0 --policy_lr 3e-4 --value_lr 1e-3 --num_iter 100 --batch_size 64 --eval_mode
CUDA_VISIBLE_DEVICES=0 python3.6 grid_search_vectorized.py --exe ~/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --logdir final-baseline-eval --n_env_seeds 2 --n_policy_seeds 3 --ff_weight 0.5 --penalty_weight 0.0 --policy_lr 3e-4 --value_lr 1e-3 --num_iter 100 --batch_size 64 --eval_mode

CUDA_VISIBLE_DEVICES=0 python3.6 grid_search_vectorized.py --exe ~/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --logdir final-baseline-eval --n_env_seeds 2 --n_policy_seeds 3 --ff_weight 0.5 --penalty_weight 1.0 --policy_lr 3e-4 --value_lr 1e-3 --num_iter 100 --batch_size 64 --eval_mode
CUDA_VISIBLE_DEVICES=0 python3.6 grid_search_vectorized.py --exe ~/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --logdir final-baseline-eval --n_env_seeds 2 --n_policy_seeds 3 --ff_weight 1.0 --penalty_weight 0.5 --policy_lr 3e-4 --value_lr 1e-3 --num_iter 100 --batch_size 64 --eval_mode

CUDA_VISIBLE_DEVICES=0 python3.6 grid_search_vectorized.py --exe ~/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --logdir final-baseline-eval --n_env_seeds 2 --n_policy_seeds 3 --ff_weight 1.0 --penalty_weight 1.0 --policy_lr 3e-4 --value_lr 1e-3 --num_iter 100 --batch_size 64 --eval_mode

CUDA_VISIBLE_DEVICES=0 python3.6 grid_search_vectorized.py --exe ~/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --logdir final-baseline-eval --n_env_seeds 2 --n_policy_seeds 3 --ff_weight 0.5 --penalty_weight 0.5 --policy_lr 3e-4 --value_lr 5e-4 --num_iter 100 --batch_size 64 --eval_mode
CUDA_VISIBLE_DEVICES=0 python3.6 grid_search_vectorized.py --exe ~/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --logdir final-baseline-eval --n_env_seeds 2 --n_policy_seeds 3 --ff_weight 0.5 --penalty_weight 0.5 --policy_lr 5e-5 --value_lr 1e-3 --num_iter 100 --batch_size 64 --eval_mode
echo All done
