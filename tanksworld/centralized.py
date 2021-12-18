import my_config as cfg
from centralized_util import CentralizedTraining
import numpy as np

if __name__ == '__main__':  
    args = cfg.args
    params = vars(args)
    #env_params = {"exe": args.exe, "training_tanks": [0],"static_tanks":[], "random_tanks":[5,6,7,8,9], "disable_shooting":[1,2,3,4,5,6,7,8,9]}
    training_tank = params['n_tank_train']
    if params['config'] == 1:
        params['env_params'] = {"exe": args.exe, "training_tanks": [0],"static_tanks":[1,2,3,4,5,6,7,8,9], "random_tanks":[], 
                'friendly_fire':True, 'take_damage_penalty':True, 'kill_bonus':True, 'death_penalty':False,
                "disable_shooting":[1,2,3,4,5,6,7,8,9], 'penalty_weight': params['penalty_weight'], 'timeout': params['env_timeout']}
        params['config_desc'] = 'static,no-shoot'
    elif params['config'] == 2:
        params['env_params'] = {"exe": args.exe, "training_tanks": [0],"static_tanks":[1,2,3,4,6,7,8,9], "random_tanks":[5], 
                'friendly_fire':False, 'take_damage_penalty':True, 'kill_bonus':True, 'death_penalty':False,
                "disable_shooting":[1,2,3,4,6,7,8,9], 'penalty_weight': params['penalty_weight'], 'timeout': params['env_timeout']}
        params['config_desc'] = 'random-tank-5'
    elif params['config'] == 3:
        params['env_params'] = {"exe": args.exe, "training_tanks": [0, 1],"static_tanks":[2,3,4,6,7,8,9], "random_tanks":[5], 
                'friendly_fire':True, 'take_damage_penalty':True, 'kill_bonus':True, 'death_penalty':False,
                "disable_shooting":[1,2,3,4,6,7,8,9], 'penalty_weight': params['penalty_weight'], 'timeout': params['env_timeout']}
        params['config_desc'] = 'train2-tanks,static,no-shoot-random-[5]'
    elif params['config'] == 4:
        params['env_params'] = {"exe": args.exe, "training_tanks": [0, 1],"static_tanks":[2,3,4,6,7,8,9], "random_tanks":[5], 
                'friendly_fire':True, 'take_damage_penalty':True, 'kill_bonus':True, 'death_penalty':False,
                "disable_shooting":[1,2,3,4,6,7,8,9], 'penalty_weight': params['penalty_weight'], 'timeout': params['env_timeout']}
        params['config_desc'] = 'train2-tanks,static,no-shoot-random-[5]'

    centralized_training = CentralizedTraining(**params)
    if args.record:
        centralized_training.record(args.video_path)
    elif args.training:
        centralized_training.train()
    elif args.eval_mode:
        centralized_training.eval()

    #else:#training
