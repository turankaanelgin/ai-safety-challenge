import my_config as cfg
from centralized_util import CentralizedTraining
import numpy as np
from env_config import update_env_config

if __name__ == "__main__":
    args = cfg.args
    params = vars(args)
    update_env_config(params)
    centralized_training = CentralizedTraining(**params)
    if args.record:
        centralized_training.record(args.video_path)
    elif args.training:
        centralized_training.train()
    elif args.eval_mode:
        centralized_training.eval()

    # else:#training
