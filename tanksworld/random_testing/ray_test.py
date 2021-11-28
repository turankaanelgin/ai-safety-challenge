# Import the RL algorithm (Trainer) we would like to use.
from ray.rllib.agents.ppo import PPOTrainer
from rgb_env import RGBCartPole
import gym
from ray.rllib.agents.trainer import Trainer, with_common_config
#env = gym.make('Pong-v0')
#print(env.reset().shape)
#env = gym.make("RGBCartPole-v0")
#import pdb; pdb.set_trace();

# Configure the algorithm.
#config = with_common_config({
config = {
        # Environment (RLlib understands openAI gym registered strings).
        "env_config": {'width':150, 'height': 100},
        "num_gpus": 1,
        "record_env": True,
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE (lambda) parameter.
        "lambda": 1.0,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.2,
        # Size of batches collected from each worker.
        "rollout_fragment_length": 200,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 4000,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 128,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 30,
        # Stepsize of SGD.
        "lr": 5e-5,
        # Learning rate schedule.
        "lr_schedule": None,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers=True inside your model's config.
        "vf_loss_coeff": 1.0,
        "model": {

            "conv_filters": 
                [
                        [16, [8, 8], 4],
                        [32, [4, 4], 2],
                        [128, [13, 19], 1],
                    ],

            #"conv_filters": None,
            #[
            #    [16, [8, 8], 4],
            #    [42, [4, 4], 2],
            #    [64, [4, 4], 1],
            #],
            # Activation function descriptor.
            # Supported values are: "tanh", "relu", "swish" (or "silu"),
            # "linear" (or None).
            "conv_activation": "relu",
            "vf_share_layers": False,
            #"no_final_linear": True,
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
            },
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.0,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.3,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None, # Target value for KL divergence.
        "kl_target": 0.01,
# Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "truncate_episodes",
# Which observation filter to apply to the observation.
        "observation_filter": "NoFilter", 
# Deprecated keys:
# Share layers for value function. If you set this to True, it's important
# to tune vf_loss_coeff.
# Use config.model.vf_share_layers instead.
        "num_workers": 0,
        "num_envs_per_worker": 1, 
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        # Set up a separate evaluation worker set for the
        # `trainer.evaluate()` call after training (see below).
        #"evaluation_num_workers": 1,
        # Only for evaluation runs, render the env.
        #"evaluation_config": {
        #    "render_env": True,
        #    }
        }

# Create our RLlib Trainer.
trainer = PPOTrainer(env=RGBCartPole, config=config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(100):
    print(trainer.train())

    # Evaluate the trained Trainer (and render each timestep to the shell's
    # output).
    #trainer.evaluate()
