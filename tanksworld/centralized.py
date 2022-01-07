import my_config as cfg
from centralized_util import CentralizedTraining
import numpy as np

if __name__ == '__main__':  
    args = cfg.args
    params = vars(args)
    #env_params = {"exe": args.exe, "training_tanks": [0],"static_tanks":[], "random_tanks":[5,6,7,8,9], "disable_shooting":[1,2,3,4,5,6,7,8,9]}
    training_tank = params['n_tank_train']
    if params['config'] == 1:
        params['env_params'] = {"exe": args.exe, 'friendly_fire':True, 'take_damage_penalty':True, 'kill_bonus':True, 'death_penalty':False,
                "training_tanks": [0],"static_tanks":[1,2,3,4,6,7,8,9], "random_tanks":[5], 
                'enable_input_tanks': [0, 1, 2, 3, 4], 'enable_output_tanks': [0, 1, 2, 3, 4], 'input_type': params['input_type'], 
                "disable_shooting":[1,2,3,4,6,7,8,9], 'penalty_weight': params['penalty_weight'], 'timeout': params['env_timeout']}
        params['config_desc'] = '1vs1-freeze-all-5input-5out'
    elif params['config'] == 2:
        params['env_params'] = {"exe": args.exe, 'friendly_fire':True, 'take_damage_penalty':True, 'kill_bonus':True, 'death_penalty':False,
                "training_tanks": [0],"static_tanks":[1,2,3,4,6,7,8,9], "random_tanks":[5], 
                'enable_input_tanks': [0], 'enable_output_tanks': [0], 'input_type': params['input_type'], 
                "disable_shooting":[1,2,3,4,6,7,8,9], 'penalty_weight': params['penalty_weight'], 'timeout': params['env_timeout']}
        params['config_desc'] = '1vs1-1input-1output'
    elif params['config'] == 3:
        params['env_params'] = {"exe": args.exe, 'friendly_fire':True, 'take_damage_penalty':True, 'kill_bonus':True, 'death_penalty':False,
                "training_tanks": [0,1],"static_tanks":[2,3,4,6,7,8,9], "random_tanks":[5], 
                'enable_input_tanks': [0,1], 'enable_output_tanks': [0,1], 'input_type': params['input_type'], 
                "disable_shooting":[2,3,4,6,7,8,9], 'penalty_weight': params['penalty_weight'], 'timeout': params['env_timeout']}
        params['config_desc'] = '2vs1-2input-2output'
    elif params['config'] == 4:
        params['env_params'] = {"exe": args.exe, 'friendly_fire':True, 'take_damage_penalty':True, 'kill_bonus':True, 'death_penalty':False,
                "training_tanks": [0,1,2],"static_tanks":[3,4,6,7,8,9], "random_tanks":[5], 
                'enable_input_tanks': [0,1,2], 'enable_output_tanks': [0,1,2], 'input_type': params['input_type'], 
                "disable_shooting":[3,4,6,7,8,9], 'penalty_weight': params['penalty_weight'], 'timeout': params['env_timeout']}
        params['config_desc'] = '3vs1-3input-3output'
    elif params['config'] == 5:
        params['env_params'] = {"exe": args.exe, 'friendly_fire':True, 'take_damage_penalty':True, 'kill_bonus':True, 'death_penalty':False,
                "training_tanks": [0,1,2,3],"static_tanks":[4,6,7,8,9], "random_tanks":[5], 
                'enable_input_tanks': [0,1,2,3], 'enable_output_tanks': [0,1,2,3], 'input_type': params['input_type'], 
                "disable_shooting":[4,6,7,8,9], 'penalty_weight': params['penalty_weight'], 'timeout': params['env_timeout']}
        params['config_desc'] = '4vs1-4input-4output'
    elif params['config'] == 6:
        params['env_params'] = {"exe": args.exe, 
                'friendly_fire':True, 'take_damage_penalty':True, 'kill_bonus':False, 'death_penalty':False, 
                "training_tanks": [0],"static_tanks":[1,2,3,4,6,7,8,9], "random_tanks":[5], "disable_shooting":[1,2,3,4,6,7,8,9],
                'enable_input_tanks': [0], 'enable_output_tanks': [0],
                'input_type': params['input_type'], 
                'penalty_weight': params['penalty_weight'], 'timeout': params['env_timeout']}
        params['config_desc'] = '1vs1'
    elif params['config'] == 7:
        params['env_params'] = {"exe": args.exe, 
                'friendly_fire':True, 'take_damage_penalty':True, 'kill_bonus':False, 'death_penalty':False, 
                "training_tanks": [0, 1],"static_tanks":[2,3,4,6,7,8,9], "random_tanks":[5], "disable_shooting":[2,3,4,6,7,8,9],
                'enable_input_tanks': [0, 1], 'enable_output_tanks': [0, 1], 'input_type': params['input_type'], 
                'penalty_weight': params['penalty_weight'], 'timeout': params['env_timeout']}
        params['config_desc'] = '2vs1'
    elif params['config'] == 8:
        params['env_params'] = {"exe": args.exe, 
                'friendly_fire':True, 'take_damage_penalty':True, 'kill_bonus':True, 'death_penalty':False, 
                "training_tanks": [0, 1, 2, 3, 4],"static_tanks":[6,7,8,9], "random_tanks":[5], "disable_shooting":[6,7,8,9],
                'enable_input_tanks': [0, 1, 2, 3, 4], 'enable_output_tanks': [0, 1, 2, 3, 4], 'input_type': params['input_type'], 
                'penalty_weight': params['penalty_weight'], 'timeout': params['env_timeout']}
        params['config_desc'] = '5vs1'
    elif params['config'] == 9:
        params['env_params'] = {"exe": args.exe, 
                'friendly_fire':True, 'take_damage_penalty':True, 'kill_bonus':False, 'death_penalty':False, 
                "training_tanks": [0, 1, 2, 3, 4], "static_tanks":[], "random_tanks":[5,6,7,8,9], "disable_shooting":[],
                'enable_input_tanks': [0, 1, 2, 3, 4], 'enable_output_tanks': [0, 1, 2, 3, 4], 'input_type': params['input_type'], 
                'penalty_weight': params['penalty_weight'], 'timeout': params['env_timeout']}
        params['config_desc'] = '5vs5'
    centralized_training = CentralizedTraining(**params)
    if args.record:
        centralized_training.record(args.video_path)
    elif args.training:
        centralized_training.train()
    elif args.eval_mode:
        centralized_training.eval()

    #else:#training
