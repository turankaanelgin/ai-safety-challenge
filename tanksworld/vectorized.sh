#!/bin/bash
train(){
    python vectorized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --horizon 64 --n-env 10 --penalty-weight $1 --save-freq 10000 --timestep 2000000 \
        --desc $2 
}

record(){
    python vectorized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --horizon 64 --n-env 10 --penalty_weight 0.2 --save-freq 10000 --timestep 4000000 \
        --desc "hor64-penalty-0.8-larger-model-separate-arch-20-envs"\
        --save-path ./results/21-10-27-20:11-hor64-penalty-0.8-larger-model/checkpoints/rl_model_900000_steps.zip\
        --record
}

record_rgb(){
    python vectorized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --horizon 64 --n-env 10 --penalty-weight 0.2 --save-freq 10000 --timestep 4000000 \
        --desc "hor64-penalty-0.8-larger-model-separate-arch-20-envs"\
        --save-path results/21-11-02-15:21-hor64-penalty-0.2-larger-model-separate-arch-10-envs/checkpoints/rl_model_400000_steps.zip \
        --record-rgb

        #results/21-10-31-13:55-hor64-penalty-0.6-larger-model-separate-arch-10-envs/checkpoints/rl_model_1900000_steps.zip\
}
env_exp(){
    python env_experiments.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --horizon 64 --n-env 10 --penalty-weight 0.2 --save-freq 10000 --timestep 4000000 \
        --desc "hor64-penalty-0.8-larger-model-separate-arch-20-envs"\
        --save-path ./results/21-10-27-20:11-hor64-penalty-0.8-larger-model/checkpoints/rl_model_900000_steps.zip\
        --record
}

test(){
    python vectorized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --horizon 64 --n-env 1 --penalty-weight 0.2 --save-freq 10000 --timestep 4000000 \
        --desc 'some testing' --testing
}
ppo_test(){
    python ppo_test.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --horizon 64 --n-env 5 --penalty-weight 0.2 --save-freq 10000 --timestep 4000000 \
        --desc 'some testing' --testing
}
if [[ $1 == test ]]; then
    test
elif [[ $1 == ppotest ]]; then
    ppo_test
elif [[ $1 == envexp ]]; then
    env_exp
elif [[ $1 == train ]]; then
    train 0.0 "hor64-penalty-0.0-larger-model-separate-arch-10-envs"
    train 0.05 "hor64-penalty-0.05-larger-model-separate-arch-10-envs"
elif [[ $1 == train04 ]]; then
    train 0.4 "hor64-penalty-0.4-larger-model-separate-arch-10-envs"
elif [[ $1 == train02 ]]; then
    train 0.2 "hor64-penalty-0.2-larger-model-separate-arch-10-envs"
elif [[ $1 == record ]]; then
    record
elif [[ $1 == record_rgb ]]; then
    record_rgb
elif [[ $1 == process ]]; then
    ps aux | grep ai-safety-challenge
elif [[ $1 == pkill ]]; then
    pkill -f "ai-safety-challenge"
elif [[ $1 == gentag ]]; then
    ctags -R . /home/ado8/stable-baselines3 /home/ado8/ai-arena/arena5/
else
    echo "no matching command"
fi


