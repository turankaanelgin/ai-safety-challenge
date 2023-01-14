if [[ $1 == train ]]; then
    python trainer.py --logdir test --exe ../exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --policy_lr 3e-4 --value_lr 1e-3 \
        --init_log_std -0.5 --num_workers 8 --num_iter 1000000 --centralized --centralized_critic --reward_weight 1.0 --penalty_weight 0.5 --batch_size 64
elif [[ $1 == debug ]]; then
    python trainer.py --logdir test --exe ../exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --policy_lr 3e-4 --value_lr 1e-3 \
        --init_log_std -0.5 --num_workers 2 --num_iter 20000000 --centralized --centralized_critic --reward_weight 1.0 --penalty_weight 0.0 --batch_size 512 \
        --use_state_vector  --save_tag tank-vector-bounding-box-512x6-layers --rollout_length 2048 --hidden_size 512 512 512 512 512 512 --attach_bounding_box
elif [[ $1 == tank-vector ]]; then
    python trainer.py --logdir test --exe ../exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --policy_lr 3e-4 --value_lr 1e-3 \
        --init_log_std -0.5 --num_workers 30 --num_iter 20000000 --centralized --centralized_critic --reward_weight 1.0 --penalty_weight 0.0 --batch_size 4096 \
        --use_state_vector  --save_tag state-vec-bb-largebatch-small-model --rollout_length 40960 --optimzation_epochs 30 --hidden_size 512 512 64 64 --attach_bounding_box
elif [[ $1 == train-lunar ]]; then
    python trainer.py --logdir test --exe ../exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --policy_lr 3e-4 --value_lr 1e-3 \
        --init_log_std -0.5 --num_workers 16 --num_iter 10000000 --centralized --centralized_critic --reward_weight 1.0 --penalty_weight 0.3 --batch_size 64 \
        --use_state_vector --env_name LunarLanderContinuous-v2 --single_agent --save_tag lunar4 --rollout_length 2048 --hidden_size 512 512
elif [[ $1 == train-vec-walker ]]; then
    python trainer.py --logdir test --exe ../exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --policy_lr 3e-4 --value_lr 1e-3 \
        --init_log_std -0.5 --num_workers 1 --num_iter 10 --centralized --reward_weight 1.0 --penalty_weight 0.3 --batch_size 32 \
        --visual_mode --use_state_vector --env_name BipedalWalker-v3 --single_agent --save_tag walker2
elif [[ $1 == record ]]; then
    python trainer.py --logdir test --exe ../exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --policy_lr 3e-4 --value_lr 1e-3 \
        --init_log_std -0.5 --num_workers 1 --num_iter 50 --centralized --centralized_critic --reward_weight 1.0 --penalty_weight 0.0 --batch_size 2048 \
        --use_state_vector  --save_tag tank-vector-bounding-box-larg-batch --rollout_length 2048 --hidden_size 2048 2048 2048 512 128 --attach_bounding_box --visual_mode
elif [[ $1 == record-lunar ]]; then
    python trainer.py --logdir test --exe ../exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --policy_lr 3e-4 --value_lr 1e-3 \
        --init_log_std -0.5 --num_workers 1 --num_iter 10000000 --centralized --centralized_critic --reward_weight 1.0 --penalty_weight 0.3 --batch_size 64 \
        --use_state_vector --env_name LunarLanderContinuous-v2 --single_agent --save_tag lunar2 --visual_mode
elif [[ $1 == record1 ]]; then
    python trainer.py --logdir test --eval_logdir test --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 --teamname1 red --teamname2 blue --reward_weight 1.0 --penalty_weight 0.5 --ff_weight 0.0 --curriculum_stop -1 --policy_lr 3e-4 --value_lr 0.001 --entropy_coef 0.0 --batch_size 64 --num_epochs 4 --clip_ratio 0.2 --friendly_fire --num_iter 1000000 --num_workers 10 --freeze_rep --init_log_std -0.5 --n_env_seeds 1 --n_policy_seeds 1 --seed_idx 0 --n_eval_seeds 10 --num_eval_episodes 100 --visual_mode --centralized
elif [[ $1 == getimg ]]; then
#   scp "bernese:/home/ado8/ai-safety-challenge/tanksworld/logs/test/lrp=0.0003__lrv=0.001__r=1.0__p=0.0__H=64__ROLLOUT=10/seed0/*.png" .
   scp "bernese:/home/ado8/ai-safety-challenge/tanksworld/logs/test/lrp=0.0003__lrv=0.001__r=1.0__p=0.0__H=64__ROLLOUT=10__large_model/seed0/*.png" .
else
    echo "not a command"
fi
