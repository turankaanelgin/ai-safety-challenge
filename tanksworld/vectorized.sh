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

test(){
    python vectorized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --horizon 64 --n-env 1 --penalty-weight 0.2 --save-freq 10000 --timestep 4000000 \
        --desc 'some testing' --testing
}
if [[ $1 == test ]]; then
    test
elif [[ $1 == train04 ]]; then
    train 0.4 "hor64-penalty-0.4-larger-model-separate-arch-10-envs"
elif [[ $1 == train02 ]]; then
    train 0.2 "hor64-penalty-0.2-larger-model-separate-arch-10-envs"
elif [[ $1 == record ]]; then
    record
elif [[ $1 == process ]]; then
    ps aux | grep ai-safety-challenge
elif [[ $1 == pkill ]]; then
    pkill -f "ai-safety-challenge"
elif [[ $1 == gentag ]]; then
    ctags -R . /home/ado8/stable-baselines3 /home/ado8/ai-arena/arena5/
fi


