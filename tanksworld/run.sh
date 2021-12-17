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

