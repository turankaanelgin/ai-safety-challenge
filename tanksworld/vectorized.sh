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
        --record-episodes 5 \
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
garage(){
    xvfb-run -s "-screen 0 1280x1024x24" --auto-servernum --server-num=1 python garage_test.py
}

envexp(){
    xvfb-run -s "-screen 0 1280x1024x24" --auto-servernum --server-num=1 python env_experiments.py \
        --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --horizon 128 --n-env 8 --penalty-weight 0.2 --save-freq 100000 --timestep 300000 \
        --desc "pole-rgb"\
        --testing
        #--save-path testing/21-11-18-21:40:33-lunar-planer-rgb-horizon-1024-env-4/checkpoints/rl_model_1200000_steps.zip\
        #--debug --testing
        #--record-rgb \
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
    envexp
elif [[ $1 == train ]]; then
    train 0.1 "hor64-penalty-0.1-larger-model-separate-arch-10-envs"
    train 0.15 "hor64-penalty-0.15-larger-model-separate-arch-10-envs"
    train 0.2 "hor64-penalty-0.2-larger-model-separate-arch-10-envs"
    train 0.25 "hor64-penalty-0.25-larger-model-separate-arch-10-envs"
elif [[ $1 == train04 ]]; then
    train 0.4 "hor64-penalty-0.4-larger-model-separate-arch-10-envs"
elif [[ $1 == train02 ]]; then
    train 0.2 "hor64-penalty-0.2-larger-model-separate-arch-10-envs"
elif [[ $1 == record ]]; then
    record
elif [[ $1 == garage ]]; then
    garage
elif [[ $1 == record_rgb ]]; then
    record_rgb
elif [[ $1 == process ]]; then
    ps aux | grep ai-safety-challenge
elif [[ $1 == pkill ]]; then
    pkill -f "ai-safety-challenge"
elif [[ $1 == gentag ]]; then
    ctags -R --languages=python --python-kinds=-i .  /home/ado8/rgb-env /home/ado8/ai-arena/arena5/  /home/ado8/rl-baselines3-zoo /home/ado8/stable-baselines3 /home/ado8/ray
#/home/ado8/garage/src/ 
elif [[ $1 == lab ]]; then
    xvfb-run -s "-screen 0 1280x1024x24" jupyter lab
else
    echo "no matching command"
fi

