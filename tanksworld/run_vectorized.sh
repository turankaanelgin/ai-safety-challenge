#!/bin/bash
train(){
    python vectorized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --horizon 64 --n-env 10 --penalty_weight $1 --save-freq 10000 --timestep 4000000 \
        --desc $2 
}

test(){
    python vectorized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --horizon 64 --n-env 1 --penalty_weight 0.2 --save-freq 10000 --timestep 4000000 \
        --desc 'some testing' --testing
}
if [[ $1 == test ]]; then
    test
elif [[ $1 == train ]]; then
    train 0.2 "hor64-penalty-0.2-larger-model-separate-arch-10-envs"
    train 0.4 "hor64-penalty-0.4-larger-model-separate-arch-10-envs"
    train 0.6 "hor64-penalty-0.6-larger-model-separate-arch-10-envs"
fi


