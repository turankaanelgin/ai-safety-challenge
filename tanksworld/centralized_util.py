from collections import deque
from os.path import join as pjoin
import optuna
import pickle
import json
from typing import Callable
import numpy as np
from stable_baselines3 import PPO
import gym
import cv2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import torch.nn as nn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
    VecEnvWrapper,
)
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import torch
import os
import yaml
from collections import deque
from typing import Callable
import numpy as np
from stable_baselines3 import PPO
from matplotlib import pyplot as plt
import gym
import cv2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import torch.nn as nn
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
    VecEnvWrapper,
)
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from tanksworld.env_centralized.env import TanksWorldEnv
import torch
import os
import yaml
from datetime import datetime
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
)
from tensorboardX import SummaryWriter

from tanksworld.env_centralized.minimap_util import displayable_rgb_map


class CentralizedTraining:
    def __init__(self, **params):
        self.params = params
        preload_str = "preloaded" if params["save_path"] is not None else ""
        load_type_str = params["load_type"] if params["save_path"] is not None else ""
        freeze_cnn_str = "freeze-cnn" if params["freeze_cnn"] else ""
        if params["debug"]:
            self.save_path = pjoin(
                self.params["experiment"], "debug", params["exp_desc"]
            )
        else:
            self.save_path = pjoin(
                self.params["experiment"], "train", params["exp_desc"]
            )
            if params["continue_training"]:
                self.save_path = params["save_path"]

        os.makedirs(self.save_path, exist_ok=True)
        self.model_path = params["model_path"]

        if (
            self.model_path is None
            and self.params["save_path"] is not None
            and self.params["model_num"] > 0
        ):
            self.model_path = pjoin(
                self.params["save_path"],
                "checkpoints",
                "rl_model_{}_steps.zip".format(self.params["model_num"]),
            )

        if self.params["training"]:
            self.training_env = self.create_env(self.params["n_envs"])
            self.training_model = self.create_model()
        elif self.params["record"]:
            # check_env(self.training_env)
            self.eval_env = TanksWorldEnv(**self.params["env_params"], will_render=True)
            self.eval_model = PPO.load(self.model_path, env=self.eval_env)
            # check_env(self.env)

    def create_env(self, n_envs):
        def create_env_():
            return TanksWorldEnv(**self.params["env_params"])

        if self.params["dummy_proc"]:
            env = make_vec_env(create_env_, n_envs=n_envs, vec_env_cls=DummyVecEnv)
        else:
            env = make_vec_env(create_env_, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
        # env = VecFrameStack(env, 4)
        # wrap env into a Monitor
        n_training_tank = len(self.params["env_params"]["training_tanks"])
        env = VecEnvTankworldMonitor(env, n_envs, n_training_tank, self.save_path)
        return env

    def create_model(self):
        model_path = self.model_path
        if self.params["continue_training"]:
            print("load model {}".format(model_path))
            assert model_path is not None
            model = PPO.load(model_path, env=self.training_env)
        else:
            policy_kwargs = {}

            policy_type = None
            if self.params["input_type"] == "stacked":
                policy_type = "CnnPolicy"
                features_extractor_class = CustomCNN
            elif self.params["input_type"] == "dict":
                policy_type = "MultiInputPolicy"
                features_extractor_class = CustomDictExtractor

            policy_kwargs = dict(
                features_extractor_class=features_extractor_class,
                features_extractor_kwargs={
                    "model_type": self.params["extract_ftr_model"],
                    "features_dim": self.params["features_dim"],
                },
                net_arch=[
                    dict(
                        pi=[self.params["net_arch_size"]],
                        vf=[self.params["net_arch_size"]],
                    )
                ],
            )

            def linear_schedule(initial_value: float) -> Callable[[float], float]:
                def func(progress_remaining: float) -> float:
                    return progress_remaining * initial_value

                return func

            if self.params["learning_rate_type"] == "linear":
                learning_rate = linear_schedule(self.params["learning_rate"])
            elif self.params["learning_rate_type"] == "constant":
                learning_rate = self.params["learning_rate"]

            model = PPO(
                policy_type,
                self.training_env,
                policy_kwargs=policy_kwargs,
                n_steps=self.params["n_steps"],
                learning_rate=learning_rate,
                verbose=0,
                batch_size=self.params["batch_size"],
                ent_coef=self.params["ent_coef"],
                clip_range=self.params["clip_range"],
                n_epochs=self.params["epochs"],
                tensorboard_log=self.save_path,
            )
            if self.params["load_type"] == "cnn":
                print("load model {}".format(model_path))
                assert model_path is not None
                loaded_model = PPO.load(model_path)
                model.policy.features_extractor.load_state_dict(
                    loaded_model.policy.features_extractor.state_dict()
                )
            elif self.params["load_type"] == "full":
                print("load model {}".format(model_path))
                assert model_path is not None
                loaded_model = PPO.load(model_path)
                model.policy.load_state_dict(loaded_model.policy.state_dict())

            if self.params["freeze_cnn"]:
                for param in model.policy.features_extractor.parameters():
                    param.requires_grad = False
        return model

    def record(self, save_video_path):
        def remove_frame(ax):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        observation = self.eval_env.reset()
        episode = 1
        observation_list = []
        win, lose = 0, 0
        while episode <= self.params["n_episode"]:
            action, _ = self.eval_model.predict(observation)
            observation, reward, done, info = self.eval_env.step(action)
            if self.params["input_type"] == "stacked":
                observation0 = observation.transpose(1, 2, 0)
                tanks_img = displayable_rgb_map(observation0)
                fig, axes = plt.subplots(1, 1)
                plt.imshow(tanks_img)
                fig.canvas.draw()
                data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close()
                observation_list.append(data)
            elif self.params["input_type"] == "dict":
                observation, reward, done, info = self.eval_env.step(action)
                # use matplotlib to draw image
                fig, axes = plt.subplots(1, 2)
                remove_frame(axes[0])
                remove_frame(axes[1])
                ally_dmg_inflict = info["red_stats"]["damage_inflicted_on"]
                enemy_dmg_inflict = info["blue_stats"]["damage_inflicted_on"]
                desc = (
                    "Episode: {}\nAlly:\ndmg to enemies: {}\ndmg to allies: {}\n"
                    + "---------------------------------\n"
                    + "Enemy:\ndmg to enemies: {}\ndmg to allies: {}\n"
                    + "---------------------------------\n"
                    + "Win: {}\nLose: {}"
                ).format(
                    episode,
                    round(ally_dmg_inflict["enemy"], 2),
                    round(ally_dmg_inflict["ally"], 2),
                    round(enemy_dmg_inflict["enemy"], 2),
                    round(enemy_dmg_inflict["ally"], 2),
                    win,
                    lose,
                )
                axes[0].imshow(self.eval_env.overviewmap)
                axes[1].text(0, 0.5, desc)
                fig.canvas.draw()
                data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close()
                observation_list.append(data)
            if done:
                # add delay before moving in the next episode
                for _ in range(4):
                    observation_list.append(data)
                dmg_for_ally = ally_dmg_inflict["enemy"] - ally_dmg_inflict["ally"]
                dmg_for_enemy = enemy_dmg_inflict["enemy"] - enemy_dmg_inflict["ally"]
                if dmg_for_ally > dmg_for_enemy:
                    win += 1
                else:
                    lose += 1
                episode += 1
                observation = self.eval_env.reset()
        out = cv2.VideoWriter(
            save_video_path, cv2.VideoWriter_fourcc(*"MJPG"), 3, (640, 480), True
        )
        for img in observation_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out.write(img)
        out.release()

    def train(self):
        #        if not os.path.exists(self.save_path):
        #            os.mkdir(self.save_path)
        with open(self.save_path + "/config.yaml", "w") as f:
            yaml.dump(self.params, f)

        checkpoint_callback = CheckpointCallback(
            save_freq=self.params["save_freq"],
            save_path=self.save_path + "/checkpoints",
            name_prefix="rl_model",
        )
        tensorboard_callback = TankworldLoggerCallback()
        trial_eval_callback = TrialEvalCallback(self.params["trial"])
        early_stop_callback = EarlyStopCallback(
            n_training_agent=len(self.params["env_params"]["training_tanks"])
        )
        #        callback_list = []
        callback_list = [
            checkpoint_callback,
            tensorboard_callback,
            early_stop_callback,
            trial_eval_callback,
        ]
        self.training_model.learn(
            total_timesteps=self.params["timestep"],
            callback=callback_list,
            reset_num_timesteps=not self.params["continue_training"],
        )
        self.score = self.training_env.get_score()
        self.training_env.close()

    def eval(self):
        model_path = pjoin(args.save_path, "checkpoints", args.checkpoint)
        model = PPO.load(model_path)
        env = create_env()
        observation = env.reset()
        step = 0
        old_step = 0
        game = 0
        episode_statistics = []
        mean_statistics = {}
        std_statistics = {}
        all_statistics = {}
        while True:
            action, _ = model.predict(observation)
            # observation, reward, done, info = env.step(np.random.rand(15))
            observation, reward, done, info = env.step(action)
            step += 1
            if done:
                info["average"]["step"] = step - old_step
                episode_statistics.append(info["average"])
                old_step = step
                observation = env.reset()
                game += 1
                if game == args.eval_game:
                    break
        for key in episode_statistics[0]:
            list_of_stats = list(
                episode_statistics[idx][key] for idx in range(len(episode_statistics))
            )
            mean_statistics[key] = np.average(list_of_stats)
            std_statistics[key] = np.std(list_of_stats)
            all_statistics[key] = list_of_stats

        with open(pjoin(args.save_path, "mean_statistics.json"), "w+") as f:
            json.dump(mean_statistics, f, indent=True)
        with open(pjoin(args.save_path, "std_statistics.json"), "w+") as f:
            json.dump(std_statistics, f, indent=True)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space, features_dim: int = 256, input_type="stacked"
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            # nn.ReLU(),
            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            # nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CustomDictExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, model_type="small", features_dim: int = 256):
        super(CustomDictExtractor, self).__init__(observation_space, 1)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.spaces["0"].shape[0]

        base = [
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
        ]
        if model_type == "small":
            base = base + [
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
            ]
        elif model_type == "medium":
            base = base + [
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
            ]
        base.append(nn.Flatten())

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = nn.Sequential(*base)(
                th.as_tensor(observation_space.spaces["0"].sample()[None]).float()
            ).shape[1]

        linear_sequence = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.extract_module = nn.Sequential(*(list(base) + list(linear_sequence)))
        self._features_dim = features_dim * len(observation_space.spaces.items())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoded_tensor_list = []
        for key, tensor in observations.items():
            encoded_tensor_list.append(self.extract_module(tensor))
        out_ = th.cat(encoded_tensor_list, dim=1)
        return out_

        return self.linear(self.cnn(observations))


class TrialEvalCallback(BaseCallback):
    def __init__(self, trial, eval_freq=10000, n_training_tank=1, verbose=0):
        super(TrialEvalCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.trial = trial
        self.n_training_tank = n_training_tank

    def _on_rollout_end(self):
        score = self.training_env.get_score()
        self.trial.report(score, self.num_timesteps)

    def _on_step(self):
        if self.trial.should_prune():
            return False
        return True


class EarlyStopCallback(BaseCallback):
    def __init__(
        self,
        check_step=1000000,
        reward_threshold=0.1,
        enemy_damage_threshold=1,
        n_training_agent=1,
        verbose=0,
    ):
        super(EarlyStopCallback, self).__init__(verbose)
        self.check_step = check_step
        self.n_training_agent = n_training_agent
        self.enemy_damage_threshold = enemy_damage_threshold * self.n_training_agent

    def _on_training_start(self) -> None:
        self.damage_inflicted_on_enemy = self.training_env.damage_inflicted_on_enemy

    def _on_rollout_end(self):
        if (
            len(self.damage_inflicted_on_enemy) == self.damage_inflicted_on_enemy.maxlen
            and np.mean(self.damage_inflicted_on_enemy) < self.enemy_damage_threshold
        ):
            raise optuna.exceptions.TrialPruned()

    def _on_step(self) -> bool:
        #        if self.num_timesteps % 1000 == 0:
        #            if self.num_timesteps > self.check_step:
        #                return False
        #            if (
        #                len(self.training_env.prune_enemy_damage)
        #                == self.training_env.prune_enemy_damage.maxlen
        #                and np.mean(self.training_env.prune_enemy_damage)
        #                / self.n_training_agent
        #                < self.enemy_damage_threshold
        #            ):
        #                print("Early stop call, prune by enemy damage")
        #                return False
        #
        return True


class TankworldLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TankworldLoggerCallback, self).__init__(verbose)
        self.step = 0

    def _on_training_start(self) -> None:
        self.stats = self.training_env.stats

    def _on_training_end(self) -> None:
        self.training_env.save_stats()

    def _on_step(self) -> bool:
        if self.n_calls % 10000 == 0:
            self.training_env.save_stats()
        return True

    def _on_rollout_end(self) -> None:
        s = {}
        while len(self.stats) > 0:
            stats_dict = self.stats.pop()
            for key in stats_dict:
                self.logger.record_mean(key, stats_dict[key])


class VecEnvTankworldMonitor(VecEnvWrapper):
    def __init__(self, venv: VecEnv, n_env, n_training_tank, save_path):
        VecEnvWrapper.__init__(self, venv)
        self.save_path = pjoin(save_path, "stats.pickle")
        self.stats = deque(maxlen=1000)
        self.rewards = np.zeros(n_env)
        #        self.prune_enemy_damage = deque(maxlen=200)
        self.score = deque(maxlen=300)
        self.damage_inflicted_on_enemy = deque(maxlen=300)
        self.saved_stats_list = []
        self.n_training_tank = n_training_tank

    def get_score(self):
        return np.mean(self.score) if len(self.score) > 0 else 0

    def save_stats(self):
        with open(self.save_path, "wb") as handle:
            pickle.dump(self.saved_stats_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.rewards += rewards
        for i, done in enumerate(dones):
            if done:
                red_stats = infos[i]["red_stats"]
                blue_stats = infos[i]["blue_stats"]
                individual_stats = infos[i]["individual_stats"]
                self.saved_stats_list.append(
                    {
                        "red_stats": red_stats,
                        "blue_stats": blue_stats,
                        "step_per_episode": infos[i]["episode_step"],
                        "individual_stats": individual_stats,
                        "idx": len(self.saved_stats_list),
                    }
                )
                red_dmg_inflicted = red_stats["damage_inflicted_on"]
                red_dmg_taken = red_stats["damage_taken_by"]
                blue_dmg_inflicted = blue_stats["damage_inflicted_on"]
                blue_dmg_taken = blue_stats["damage_taken_by"]

                self.damage_inflicted_on_enemy.append(red_dmg_inflicted["enemy"])

                red_score = (
                    red_dmg_inflicted["enemy"]
                    - red_dmg_taken["ally"]
                    - red_dmg_taken["enemy"]
                )
                blue_score = (
                    blue_dmg_inflicted["enemy"]
                    - blue_dmg_taken["ally"]
                    - blue_dmg_taken["enemy"]
                )

                intended_score = red_dmg_inflicted["enemy"] / 100
                unintended_score = (
                    red_dmg_taken["ally"] + red_dmg_taken["enemy"]
                ) / 100
                volume_score = intended_score * (5 - unintended_score) / 25

                win = 1 if red_score > blue_score else 0
                sum_damage_score = red_score / self.n_training_tank
                sum_damage_score /= 100  # Normalize score
                #                if np.mean(self.damage_inflicted_on_enemy) < 5:
                #                    score = -2 # set low score if damange inflicted on enemy is small.
                self.score.append(volume_score)

                tsboard_log = {
                    "0_general_stats/win_rate": win,
                    "0_general_stats/step_per_episode": infos[i]["episode_step"],
                    "0_general_stats/shot_reward": red_stats["shot_reward"],
                    "0_general_stats/reward": infos[i]["episode_reward"],
                    "0_general_stats/sum_damage_score": sum_damage_score,
                    "0_general_stats/volume_score": volume_score,
                    "1_team_stats/dmg_inflict_on_enemy": red_dmg_inflicted["enemy"],
                    "1_team_stats/dmg_inflict_on_neutral": red_dmg_inflicted["neutral"],
                    "1_team_stats/dmg_inflict_on_ally": red_dmg_inflicted["ally"],
                    "1_team_stats/dmg_taken_by_ally": red_dmg_taken["ally"],
                    "1_team_stats/dmg_taken_by_enemy": red_dmg_taken["enemy"],
                    "1_team_stats/#shots": red_stats["number_shots_fired"]["ally"],
                    "2_lives/alive_ally": red_stats["tanks_alive"]["ally"],
                    "2_lives/alive_enemy": red_stats["tanks_alive"]["enemy"],
                    "2_lives/alive_neutral": red_stats["tanks_alive"]["neutral"],
                }
                # Individual stats
                for idx, tank_stats in enumerate(individual_stats):
                    tank_str = "3_tank#{}/".format(idx)
                    for key, value in tank_stats.items():
                        tank_field = "{} {}".format(tank_str, key)
                        if type(value) is dict:
                            for key1, value1 in value.items():
                                tank_field1 = "{} {}".format(tank_field, key1)
                                tsboard_log[tank_field1] = value1
                        else:
                            tsboard_log[tank_field] = value

                self.stats.append(tsboard_log)
                self.rewards[i] = 0

        return obs, rewards, dones, infos

    def close(self) -> None:
        return self.venv.close()
