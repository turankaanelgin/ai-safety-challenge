import pdb
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import os
import json

import gym
import numpy as np
import torch as th

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from algos.fast_ppo.buffers import MyRolloutBuffer

from core.policy_record import PolicyRecord


class MyOnPolicyAlgorithm(OnPolicyAlgorithm):

    def __init__(self,
                 policy: Union[str, Type[ActorCriticPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Schedule],
                 n_steps: int,
                 gamma: float,
                 gae_lambda: float,
                 ent_coef: float,
                 vf_coef: float,
                 max_grad_norm: float,
                 use_sde: bool,
                 sde_sample_freq: int,
                 policy_base: Type[BasePolicy] = ActorCriticPolicy,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 monitor_wrapper: bool = True,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True,
                 supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
            ):
        super(MyOnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            policy_base=policy_base,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=supported_action_spaces
        )

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else MyRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        self.episode_rewards = [[0.0] for _ in range(self.n_envs)]
        self.episode_length = [[0] for _ in range(self.n_envs)]

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            clipped_actions = clipped_actions.reshape(env.num_envs, 5, -1)
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos, dones, rewards)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            policy_record: PolicyRecord = None,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            self.rollout_buffer.reset()

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:

                if policy_record is not None:

                    for idx in range(len(self.episode_length[0])-1):
                        for env_idx in range(self.n_envs):
                            policy_record.add_result(self.episode_rewards[env_idx][idx],
                                                     self.episode_length[env_idx][idx])
                    policy_record.save()

                    self.episode_rewards = [[0.0] for _ in range(self.n_envs)]
                    self.episode_length = [[0] for _ in range(self.n_envs)]

            if self.num_timesteps % 100 == 0:
                save_mean_metrics, save_std_metrics = {}, {}
                mean_metrics, std_metrics = self.ep_info_buffer
                for key in mean_metrics[0]:
                    save_mean_metrics[key] = np.average([mean_metrics[env_idx][key] for env_idx in range(self.n_envs)])
                    save_std_metrics[key] = np.sqrt(np.sum([std_metrics[env_idx][key]**2 for env_idx in range(self.n_envs)]))/self.n_envs

                block_num = self.num_timesteps // 2500
                if policy_record is not None:
                    with open(os.path.join(policy_record.data_dir, 'mean_statistics_{}.json'.format(block_num)), 'w+') as f:
                        json.dump(save_mean_metrics, f, indent=True)
                    with open(os.path.join(policy_record.data_dir, 'std_statistics_{}.json'.format(block_num)), 'w+') as f:
                        json.dump(save_std_metrics, f, indent=True)

            self.train()

        callback.on_training_end()

        return self

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None,
                            rewards: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.
        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        if dones is None:
            dones = np.array([False] * len(infos))

        all_mean_metrics, all_std_metrics = [], []
        for idx, info in enumerate(infos):
            mean_metrics = info.get('mean')
            std_metrics = info.get('std')
            all_mean_metrics.append(mean_metrics)
            all_std_metrics.append(std_metrics)
            self.episode_length[idx][-1] += 1
            self.episode_rewards[idx][-1] += np.sum(rewards[idx])
            if dones[idx]:
                self.episode_length[idx].append(0)
                self.episode_rewards[idx].append(0.0)
        self.ep_info_buffer = (all_mean_metrics, all_std_metrics)