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
train_stacked(){
    python vectorized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --logdir testrun   --env-stacked\
        --ent-coef $1 --desc $2 --n-steps $3 --penalty-weight $4 --timestep $5 --n-env $6 --save-freq $7 \
        #--save-path $7
}
debug_gym(){
    xvfb-run -s "-screen 0 1280x1024x24" python test_env.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --save-freq 20000   --n-steps $1 --n-envs $2 --timestep $3 --penalty-weight $4 --training --debug
        #--save-path $7
}
debug(){
    python centralized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --save-freq 5000   --n-steps $1 --n-envs $2 --timestep $3 --penalty-weight $4 --ent-coef $5 --config $6 --lr-type constant --training \
        --debug
        #--save-path $7
}
train(){
    python centralized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --save-freq 5000   --n-steps $1 --n-envs $2 --timestep $3 --penalty-weight $4 --ent-coef $5 --config $6 --lr-type constant --training 
}
if [[ $1 == test ]]; then
    test
elif [[ $1 == train ]]; then
    train 64 20 700000 0.0 0.00 1 
    train 64 20 700000 0.4 0.00 1
    train 64 20 700000 0.8 0.00 1
    train 64 20 700000 0.0 0.00 2 
    train 64 20 700000 0.4 0.00 2
    train 64 20 700000 0.8 0.00 2
elif [[ $1 == debug ]]; then
    debug 64 10 700 0.0 0.00 1 
    debug 64 10 700 0.0 0.00 2 
elif [[ $1 == debug-dummy ]]; then
    python centralized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --save-freq 20000   --n-steps 32 --n-envs 2 --timestep 1000000 --penalty-weight 0.6 --training --debug --dummy-proc  --lr-type linear
elif [[ $1 == debug-gym ]]; then
    debug_gym 128 5 50000000  0.3
elif [[ $1 == record ]]; then
    python centralized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --n-env 1 --penalty-weight 0.2 --timestep 4000000 \
        --video-path tmp/tank_test.avi --n-episode 10 --record
elif [[ $1 == record-full-step ]]; then
    SETTING=21-12-16-17:54:41TW-nstep512-nenv10-neg-0.6
    EXPMNT=results/$SETTING
    DIRNAME=/home/ado8/ai-safety-challenge/tanksworld/$EXPMNT/checkpoints
    SAVEDIR=tmp/tank_vid/$SETTING
    mkdir -p $SAVEDIR
    for STEP in 400000
    do
        python centralized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
            --save-path $DIRNAME/rl_model_${STEP}_steps.zip \
            --video-path $SAVEDIR/$STEP.avi --n-episode 10 --record
    done
elif [[ $1 == record-full ]]; then
    EXPMNT=results/21-12-16-17:54:41TW-nstep512-nenv10-neg-0.6
    DIR_NAME=/home/ado8/ai-safety-challenge/tanksworld/$EXPMNT/checkpoints
    for FILE in $(ls $DIR_NAME)
    do
        python centralized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
            --save-path $DIR_NAME/$FILE \
            --n-env 1 --penalty-weight 0.2 --timestep 4000000 \
            --video-path tmp/tank_vid/$FILE.avi --n-episode 3 --record
    done
elif [[ $1 == train-stacked2 ]]; then
    train_stacked 0.00  "neg0.00-ent-0.00-steps512-nenvs-20-stacked-1reds" 512 0.00 600000 20 1000 1000 
    train_stacked 0.01  "neg0.00-ent-0.01-steps512-nenvs-20-stacked-1reds" 512 0.00 600000 20 1000 1000 
    train_stacked 0.00  "neg0.00-ent-0.00-steps2048-nenvs-20-stacked-1reds" 2048 0.00 600000 20 1000 1000 
    train_stacked 0.01  "neg0.00-ent-0.01-steps2048-nenvs-20-stacked-1reds" 2048 0.00 600000 20 1000 1000 

    train_stacked 0.00  "neg0.15-ent-0.00-steps512-nenvs-20-stacked-1reds" 512 0.15 600000 20 1000 1000 
    train_stacked 0.01  "neg0.15-ent-0.01-steps512-nenvs-20-stacked-1reds" 512 0.15 600000 20 1000 1000 
    train_stacked 0.00  "neg0.15-ent-0.00-steps2048-nenvs-20-stacked-1reds" 2048 0.15 600000 20 1000 1000 
    train_stacked 0.01  "neg0.15-ent-0.01-steps2048-nenvs-20-stacked-1reds" 2048 0.15 600000 20 1000 1000 
elif [[ $1 == train1 ]]; then
    #train 4 "neg0.0-small-steps8" "small" 8 0.0 4000000 "/home/ado8/ai-safety-challenge/tanksworld/results/21-11-26-10:43:28-neg0.0-small-steps8/checkpoints/rl_model_1000000_steps.zip"
    #train 4 "neg0.00-small-steps16-ent-0.000625" "small" 16 0.00 2000000 8 0.000625
    #train 4 "neg0.00-large-steps32-ent-0.0-stacked" "large" 32 0.00 2000000 8 0.0
    train 4 "neg0.00-small-steps32-ent-0.0-stacked-2reds-train" "small" 32 0.00 4000000 8 0.0
elif [[ $1 == train2 ]]; then
    #train 4 "neg0.00-small-steps16-ent-0.00125" "small" 16 0.00 2000000 8 0.00125
    train 4 "neg0.15-small-steps32-ent-0.0-stacked-2reds-train" "large" 32 0.15 4000000 8 0.0
    #train 4 "neg0.15-small-steps64-ent-0.0-stacked" "small" 64 0.15 2000000 8 0.0
    #train 4 "neg0.15-large-steps32-ent-0.0-stacked" "large" 32 0.15 2000000 8 0.0
    #train 4 "neg0.15-large-steps64-ent-0.0-stacked" "large" 64 0.15 2000000 8 0.0
elif [[ $1 == ppotest ]]; then
    ppo_test
elif [[ $1 == envexp ]]; then
    envexp
elif [[ $1 == train04 ]]; then
    train 0.4 "hor64-penalty-0.4-larger-model-separate-arch-10-envs"
elif [[ $1 == train02 ]]; then
    train 0.2 "hor64-penalty-0.2-larger-model-separate-arch-10-envs"
elif [[ $1 == garage ]]; then
    garage
elif [[ $1 == process ]]; then
    ps aux | grep ai-safety-challenge
elif [[ $1 == pkill ]]; then
    pkill -u ado8 -f "ai-safety-challenge"
elif [[ $1 == pkill1 ]]; then
    pkill -u ado8 -f "tensorboard"
elif [[ $1 == gentag ]]; then
    ctags -R --languages=python --python-kinds=-i .  /home/ado8/rgb-env /home/ado8/ai-arena/arena5/  /home/ado8/rl-baselines3-zoo /home/ado8/stable-baselines3 
    
#/home/ado8/garage/src/ /home/ado8/ray
elif [[ $1 == lab ]]; then
    xvfb-run -s "-screen 0 1280x1024x24" jupyter lab
else
    echo "no matching command"
fi

