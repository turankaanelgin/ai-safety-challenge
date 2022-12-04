
# Scripts
- **trainer_config.py** - contains configuration parameters 
- **trainer.py** - main script handling random seeds and configurations
- **env.py** - contains environment description
- **algos/torch_ppo/mappo.py** - contains MAPPO implementation and rollout buffer
- **algos/torch_ppo/core.py** - contains network models

# Arguments (These arguments are subject to change. Please check the latest config file.)
- **logdir** - Directory containing all the log files
- **exe** - Absolute path to the executable file
- **reward_weight** - Weight on damage inflicted on enemies
- **penalty_weight** - Weight on damage taken
- **curriculum_stop** - If annealing penalty weight with a curriculum, where to stop it
- **policy_lr** - Actor learning rate
- **value_lr** - Critic learning rate
- **batch_size** - Batch size
- **num_epochs** - Number of epochs for each of actor and critic training
- **clip_ratio** - Epsilon clipping parameter of PPO loss
- **num_iter** - Number of training steps
- **freeze_rep** - Freeze the representation
- **load_from_checkpoint** - Resumes the training from the checkpoint
- **num_rollout_threads** - Number of training rollout threads
- **centralized** - Centralized execution
- **independent** - Independent learning
- **init_log_std** - Initial value of actor log standard deviation
- **selfplay** - Run in self-play mode
- **n_eval_seeds** - Number of parallel environments in evaluation mode
- **num_eval_episodes** - Number of evaluation episodes
- **eval_logdir** - The directory contains the trained policy for evaluation

# Example Training

Following trains a model.

````python
python trainer.py --logdir test --exe path-to-executable --policy_lr 3e-4 --value_lr 1e-3 \
                  --init_log_std -0.5 --num_rollout_threads 10 --num_iter 1000000 \
                  --reward_weight 1.0 --penalty_weight 0.5 --batch_size 64
````

# Example Evaluation

Following evaluates the trained model.

````python
python trainer.py --logdir eval --eval_logdir test --exe path-to-executable --policy_lr 3e-4 --value_lr 1e-3 \
                --init_log_std -0.5 --num_rollout_threads 10 \
                --reward_weight 1.0 --penalty_weight 0.5 --batch_size 64 --n_eval_seeds 10 --num_eval_episodes 100
````
