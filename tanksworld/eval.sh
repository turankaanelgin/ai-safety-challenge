python vectorized.py --exe /home/ado8/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64 \
    --logdir testrun --horizon 64 --n-env 20 --penalty_weight 0.8 --save-freq 10000 --timestep 4000000 \
    --desc "hor64-penalty-0.8-larger-model-separate-arch-20-envs"\
    --save-path ./results/21-10-27-20:11-hor64-penalty-0.8-larger-model/checkpoints/rl_model_900000_steps.zip\
    --eval-mode

