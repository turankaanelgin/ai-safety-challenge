#!/bin/bash

record(){
    python vectorized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --horizon 64 --n-env 10 --penalty_weight 0.2 --save-freq 10000 --timestep 4000000 \
        --desc "hor64-penalty-0.8-larger-model-separate-arch-20-envs"\
        --save-path ./results/21-10-27-20:11-hor64-penalty-0.8-larger-model/checkpoints/rl_model_900000_steps.zip\
        --record-episodes 5 \
        --record
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
        --logdir testrun --n-steps 32 --n-env 1 --penalty-weight 0.2 --save-freq 10000 --timestep 4000000 \
        --desc 'some testing' --testing
}
ppo_test(){
    python ppo_test.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --n-steps 32 --n-env 5 --penalty-weight 0.2 --save-freq 10000 --timestep 4000000 \
        --desc 'some testing' --testing
}
record_rgb(){
    python vectorized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun  --n-env 10 --penalty-weight 0.2 --save-freq 10000 --timestep 4000000 \
        --desc "hor64-penalty-0.8-larger-model-separate-arch-20-envs"\
        --save-path $1 \
        --video-path $2  --stack-frame $3 --n-episode $4 --record-rgb
}
train(){
    python vectorized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --save-freq 20000  \
        --stack-frame $1 --desc $2 --model-size $3 --n-steps $4 --penalty-weight $5 --timestep $6 --n-env $7 --ent-coef $8\
        #--save-path $7
}
debug(){
    python vectorized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun --save-freq 20000  --env-stacked\
        --stack-frame $1 --desc $2 --model-size $3 --n-steps $4 --penalty-weight $5 --timestep $6 --n-env $7 --ent-coef $8\
        --testing
        #--save-path $7




}
if [[ $1 == test ]]; then
    test
elif [[ $1 == debug ]]; then
    debug 0 "neg0.00-small-steps32-ent-0.0-stacked-2reds-train" "large" 32 0.00 4000000 1 0.0
elif [[ $1 == train1 ]]; then
    #train 4 "neg0.0-small-steps8" "small" 8 0.0 4000000 "/home/ado8/ai-safety-challenge/tanksworld/results/21-11-26-10:43:28-neg0.0-small-steps8/checkpoints/rl_model_1000000_steps.zip"
    #train 4 "neg0.00-small-steps16-ent-0.000625" "small" 16 0.00 2000000 8 0.000625
    #train 4 "neg0.00-small-steps32-ent-0.0-stacked" "small" 32 0.00 2000000 8 0.0
    #train 4 "neg0.00-small-steps64-ent-0.0-stacked" "small" 64 0.00 2000000 8 0.0
    #train 4 "neg0.00-large-steps32-ent-0.0-stacked" "large" 32 0.00 2000000 8 0.0
    train 4 "neg0.00-small-steps32-ent-0.0-stacked-2reds-train" "small" 32 0.00 4000000 8 0.0
elif [[ $1 == train2 ]]; then
    #train 4 "neg0.00-small-steps16-ent-0.00125" "small" 16 0.00 2000000 8 0.00125
    train 4 "neg0.15-small-steps32-ent-0.0-stacked-2reds-train" "small" 32 0.15 4000000 8 0.0
    #train 4 "neg0.15-small-steps64-ent-0.0-stacked" "small" 64 0.15 2000000 8 0.0
    #train 4 "neg0.15-large-steps32-ent-0.0-stacked" "large" 32 0.15 2000000 8 0.0
    #train 4 "neg0.15-large-steps64-ent-0.0-stacked" "large" 64 0.15 2000000 8 0.0
elif [[ $1 == record-rgb-all ]]; then
    EXPMNT=21-12-06-18:25:58-neg0.15-small-steps32-ent-0.0-stacked-2reds-train
    DIR_NAME=/home/ado8/ai-safety-challenge/tanksworld/results/$EXPMNT/checkpoints
    for FILE in $(ls $DIR_NAME)
    do
        record_rgb $DIR_NAME/$FILE ./tmp/tank-vid/2reds-neg-0.15/$FILE.avi 4 10
    done
elif [[ $1 == record-rgb ]]; then
    #for step in 480000 800000 1120000 1920000
    #for step in 160000 320000 480000 
    for step in 480000 
    do
        record_rgb "/home/ado8/ai-safety-challenge/tanksworld/results/21-12-06-18:25:43-neg0.00-small-steps32-ent-0.0-stacked-2reds-train/checkpoints/rl_model_${step}_steps.zip" "./tmp/tank-vid/neg-0.15-small-2reds-${step}.avi" 4 10
    done
elif [[ $1 == ppotest ]]; then
    ppo_test
elif [[ $1 == envexp ]]; then
    envexp
elif [[ $1 == train04 ]]; then
    train 0.4 "hor64-penalty-0.4-larger-model-separate-arch-10-envs"
elif [[ $1 == train02 ]]; then
    train 0.2 "hor64-penalty-0.2-larger-model-separate-arch-10-envs"
elif [[ $1 == record ]]; then
    record
elif [[ $1 == garage ]]; then
    garage
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

