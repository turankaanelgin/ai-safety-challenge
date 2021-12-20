#!/bin/bash


garage(){
    xvfb-run -s "-screen 0 1280x1024x24" --auto-servernum --server-num=1 python garage_test.py
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
        --save-freq 5000 --n-steps $1 --n-envs $2 --timestep $3 --penalty-weight $4 --ent-coef $5 --config $6 --env-timeout $7\
        --lr 0.0001 --lr-type constant --training 
}

rm_checkpoint(){
    DIRNAME=results/$1/checkpoints
    ARR=()
    for FILE in $(ls $DIRNAME)
    do
        IFS='_' read -ra array <<< "$FILE"
        ARR+=(${array[2]})
    done
    #echo ${ARR]}|sort -r
    IFS=$'\n'
    LAST_CHECKPOINT=$(echo "${ARR[*]}"|sort -r|head -n1)
    for NUM in ${ARR[@]}
    do
        if [[ $NUM != ${LAST_CHECKPOINT} ]]; then
            trash "${DIRNAME}/rl_model_${NUM}_steps.zip"
        fi
    done
}
if [[ $1 == test ]]; then
    test
elif [[ $1 == train ]]; then
    train 64 20 7000000 0.4 0.00 2 500 
elif [[ $1 == debug ]]; then
    debug 64 10 700 0.0 0.00 1 
    debug 64 10 700 0.0 0.00 2 
elif [[ $1 == debug-dummy ]]; then
    python centralized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --save-freq 20000   --n-steps 32 --n-envs 2 --timestep 1000000 --penalty-weight 0.6 --config 2 --training --debug --dummy-proc  --lr-type linear \
        #--save-path results/21-12-17-21:51:56TW-timestep7.0M-nstep64-nenv20-timeout-500-neg-0.4-lrtype-constant-random-tank-5/checkpoints/rl_model_700000_steps.zip\
        #--freeze-cnn
elif [[ $1 == debug-gym ]]; then
    debug_gym 128 5 50000000  0.3
elif [[ $1 == record ]]; then
    python centralized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
        --n-env 1 --penalty-weight 0.2 --timestep 4000000 \
        --video-path tmp/tank_test.avi --n-episode 10 --record
elif [[ $1 == record-full-step ]]; then
    SETTING=21-12-17-10:19:42TW-timestep7.0M-nstep64-nenv20-timeout-250-neg-0.4-lrtype-constant-froze,no-shooting-tank-1->9
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
    SETTING=21-12-17-10:19:42TW-timestep7.0M-nstep64-nenv20-timeout-250-neg-0.4-lrtype-constant-froze,no-shooting-tank-1-9
    EXPMNT=results/$SETTING
    DIRNAME=/home/ado8/ai-safety-challenge/tanksworld/$EXPMNT/checkpoints
    SAVEDIR=tmp/tank_vid/$SETTING
    mkdir -p $SAVEDIR
    for FILE in $(ls $DIRNAME)
    do
        echo $FILE
        echo $SAVEDIR/$FILE.avi
        python centralized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
            --save-path $DIRNAME/$FILE \
            --n-env 1 --penalty-weight 0.2 --timestep 4000000 \
            --video-path $SAVEDIR/$FILE.avi --n-episode 10 --record
    done
elif [[ $1 == rm-checkpoints ]]; then
    for DIR in $(ls results)
    do
        rm_checkpoint $DIR
    done
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

