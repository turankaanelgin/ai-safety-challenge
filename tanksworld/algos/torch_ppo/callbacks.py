import json, os
import pdb

import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue

from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper



class EvalCallback:

    def __init__(self, env, policy_record, eval_steps=10, val_env=None, eval_env=None):
        self.env = env
        self.model = None
        self.policy_record = policy_record
        self.val_env = val_env
        self.eval_env = eval_env
        self.eval_steps = eval_steps

    def init_model(self, model):
        self.model = model

    def validate_independent_policy(self, model_state_dict, device):

        episodes = 0
        observation = self.eval_env.reset()

        self.model.load_state_dict(model_state_dict, strict=True)
        self.model.eval()

        num_envs = self.eval_env.num_envs
        ep_rr_damage = [0] * num_envs
        ep_rb_damage = [0] * num_envs
        ep_br_damage = [0] * num_envs
        curr_done = [False] * num_envs
        taken_stats = [False] * num_envs
        episode_red_blue_damages, episode_blue_red_damages = [], []
        episode_red_red_damages = []

        while episodes < self.eval_steps:

            with torch.no_grad():
                action, _, _, _ = self.model.step(torch.as_tensor(observation, dtype=torch.float32).squeeze(2).to(device))

            observation, reward, done, info = self.eval_env.step(action.cpu().numpy())
            curr_done = [done[idx] or curr_done[idx] for idx in range(num_envs)]

            for env_idx, terminal in enumerate(curr_done):
                if terminal and not taken_stats[env_idx]:
                    ep_rr_damage[env_idx] = info[env_idx]['red_stats']['damage_inflicted_on']['ally']
                    ep_rb_damage[env_idx] = info[env_idx]['red_stats']['damage_inflicted_on']['enemy']
                    ep_br_damage[env_idx] = info[env_idx]['red_stats']['damage_taken_by']['enemy']
                    taken_stats[env_idx] = True

            if np.all(curr_done):
                episode_red_red_damages.append(ep_rr_damage)
                episode_blue_red_damages.append(ep_br_damage)
                episode_red_blue_damages.append(ep_rb_damage)

                ep_rr_damage = [0] * num_envs
                ep_rb_damage = [0] * num_envs
                ep_br_damage = [0] * num_envs
                curr_done = [False] * num_envs
                taken_stats = [False] * num_envs
                episodes += 1
                observation = self.eval_env.reset()

                if episodes % 10 == 0 and episodes > 0:
                    avg_red_red_damages = np.mean(episode_red_red_damages)
                    avg_red_blue_damages = np.mean(episode_red_blue_damages)
                    avg_blue_red_damages = np.mean(episode_blue_red_damages)

                    with open(os.path.join(self.policy_record.data_dir, 'mean_eval_statistics.json'), 'w+') as f:
                        json.dump({'Number of games': episodes,
                                   'Red-Red-Damage': avg_red_red_damages.tolist(),
                                   'Red-Blue Damage': avg_red_blue_damages.tolist(),
                                   'Blue-Red Damage': avg_blue_red_damages.tolist()}, f, indent=4)

        return avg_red_blue_damages - (avg_red_red_damages + avg_blue_red_damages)


    def evaluate_policy(self, model_state_dict, device, step=0):

        episodes = 0
        observation = self.eval_env.reset()

        self.model.load_state_dict(model_state_dict, strict=True)
        self.model.eval()

        num_envs = self.eval_env.num_envs

        ep_rr_damage = [0] * num_envs
        ep_rb_damage = [0] * num_envs
        ep_br_damage = [0] * num_envs
        curr_done = [False] * num_envs
        taken_stats = [False] * num_envs
        episode_red_blue_damages, episode_blue_red_damages = [], []
        episode_red_red_damages = []
        all_episode_red_blue_damages = [[] for _ in range(num_envs)]
        all_episode_blue_red_damages = [[] for _ in range(num_envs)]
        all_episode_red_red_damages = [[] for _ in range(num_envs)]

        while episodes < self.eval_steps:
            with torch.no_grad():
                action, v, logp, _ = self.model.step(torch.as_tensor(observation, dtype=torch.float32).to(device))
            observation, reward, done, info = self.eval_env.step(action.cpu().numpy())
            curr_done = [done[idx] or curr_done[idx] for idx in range(num_envs)]

            for env_idx, terminal in enumerate(curr_done):
                if terminal and not taken_stats[env_idx]:
                    ep_rr_damage[env_idx] = info[env_idx]['red_stats']['damage_inflicted_on']['ally']
                    ep_rb_damage[env_idx] = info[env_idx]['red_stats']['damage_inflicted_on']['enemy']
                    ep_br_damage[env_idx] = info[env_idx]['red_stats']['damage_taken_by']['enemy']
                    taken_stats[env_idx] = True

            if np.all(curr_done):
                episode_red_red_damages.append(ep_rr_damage)
                episode_blue_red_damages.append(ep_br_damage)
                episode_red_blue_damages.append(ep_rb_damage)

                for env_idx in range(num_envs):
                    all_episode_red_blue_damages[env_idx].append(ep_rb_damage[env_idx])
                    all_episode_blue_red_damages[env_idx].append(ep_br_damage[env_idx])
                    all_episode_red_red_damages[env_idx].append(ep_rr_damage[env_idx])

                ep_rr_damage = [0] * num_envs
                ep_rb_damage = [0] * num_envs
                ep_br_damage = [0] * num_envs
                curr_done = [False] * num_envs
                taken_stats = [False] * num_envs
                episodes += 1
                observation = self.eval_env.reset()

                if episodes % 2 == 0 and episodes > 0:
                    avg_red_red_damages = np.mean(episode_red_red_damages)
                    avg_red_blue_damages = np.mean(episode_red_blue_damages)
                    avg_blue_red_damages = np.mean(episode_blue_red_damages)

                    with open(os.path.join(self.policy_record.data_dir, 'mean_statistics_eval_{}.json'.format(step)), 'w+') as f:
                        json.dump({'Number of games': episodes,
                                   'Red-Red Damage': avg_red_red_damages.tolist(),
                                   'Red-Blue Damage': avg_red_blue_damages.tolist(),
                                   'Blue-Red Damage': avg_blue_red_damages.tolist()}, f, indent=4)

                    with open(os.path.join(self.policy_record.data_dir, 'all_statistics_eval_{}.json'.format(step)), 'w+') as f:
                        json.dump({'Number of games': episodes,
                                   'Red-Red Damage': all_episode_red_red_damages,
                                   'Red-Blue Damage': all_episode_red_blue_damages,
                                   'Blue-Red Damage': all_episode_blue_red_damages}, f, indent=4)

                    avg_red_red_damages_per_env = np.mean(episode_red_red_damages, axis=0)
                    avg_red_blue_damages_per_env = np.mean(episode_red_blue_damages, axis=0)
                    avg_blue_red_damages_per_env = np.mean(episode_blue_red_damages, axis=0)

                    with open(os.path.join(self.policy_record.data_dir, 'mean_statistics_per_env_eval_{}.json'.format(step)),
                              'w+') as f:
                        json.dump({'Number of games': episodes,
                                   'All-Red-Red Damage': avg_red_red_damages_per_env.tolist(),
                                   'All-Red-Blue Damage': avg_red_blue_damages_per_env.tolist(),
                                   'All-Blue-Red Damage': avg_blue_red_damages_per_env.tolist()}, f, indent=4)


    def validate_policy(self, model_state_dict, device, discrete_action=False):

        steps = 0
        observation = self.val_env.reset()

        self.model.load_state_dict(model_state_dict, strict=True)
        self.model.eval()

        num_envs = self.val_env.num_envs
        ep_rr_damage = [0] * num_envs
        ep_rb_damage = [0] * num_envs
        ep_br_damage = [0] * num_envs
        curr_done = [False] * num_envs
        taken_stats = [False] * num_envs
        episode_red_blue_damages, episode_blue_red_damages = [], []
        episode_red_red_damages = []

        while steps < self.eval_steps:

            with torch.no_grad():
                pi = self.model.pi._distribution(torch.as_tensor(observation, dtype=torch.float32).to(device))
                action = pi.sample()
                if discrete_action:
                    action1 = (action // 100) * 0.04 - 1
                    action2 = ((action % 100) // 2) * 0.04 - 1
                    action3 = (action % 100) % 2 - 0.5
                    action = torch.cat((action1.unsqueeze(-1), action2.unsqueeze(-1), action3.unsqueeze(-1)), dim=-1)

            observation, reward, done, info = self.val_env.step(action.cpu().numpy())
            curr_done = [done[idx] or curr_done[idx] for idx in range(num_envs)]

            for env_idx, terminal in enumerate(curr_done):
                if terminal and not taken_stats[env_idx]:
                    ep_rr_damage[env_idx] = info[env_idx]['red_stats']['damage_inflicted_on']['ally']
                    ep_rb_damage[env_idx] = info[env_idx]['red_stats']['damage_inflicted_on']['enemy']
                    ep_br_damage[env_idx] = info[env_idx]['red_stats']['damage_taken_by']['enemy']
                    taken_stats[env_idx] = True

            if np.all(curr_done):
                episode_red_red_damages.append(ep_rr_damage)
                episode_blue_red_damages.append(ep_br_damage)
                episode_red_blue_damages.append(ep_rb_damage)

                ep_rr_damage = [0] * num_envs
                ep_rb_damage = [0] * num_envs
                ep_br_damage = [0] * num_envs
                curr_done = [False] * num_envs
                taken_stats = [False] * num_envs
                steps += 1
                observation = self.val_env.reset()

                if steps % 10 == 0 and steps > 0:
                    avg_red_red_damages = np.mean(episode_red_red_damages)
                    avg_red_blue_damages = np.mean(episode_red_blue_damages)
                    avg_blue_red_damages = np.mean(episode_blue_red_damages)

                    with open(os.path.join(self.policy_record.data_dir, 'mean_val_statistics.json'), 'w+') as f:
                        json.dump({'Number of games': steps,
                                   'Red-Red-Damage': avg_red_red_damages.tolist(),
                                   'Red-Blue Damage': avg_red_blue_damages.tolist(),
                                   'Blue-Red Damage': avg_blue_red_damages.tolist()}, f, indent=4)

        return avg_red_blue_damages - (avg_red_red_damages + avg_blue_red_damages)


    def save_metrics_multienv(self, episode_returns, episode_lengths, episode_red_blue_damages, episode_red_red_damages,
                              episode_blue_red_damages, eval_mode=False, episode_stds=None, episode_intrinsic_rewards=None):

        if len(episode_lengths) == 0: return

        if self.policy_record:
            if not eval_mode:
                for idx in range(len(episode_lengths)):
                    if episode_red_blue_damages[idx][0] is None:
                        continue
                    elif episode_stds:
                        self.policy_record.add_result(np.average(episode_returns[idx]),
                                                      np.average(episode_red_blue_damages[idx]),
                                                      np.average(episode_red_red_damages[idx]),
                                                      np.average(episode_blue_red_damages[idx]),
                                                      episode_lengths[idx],
                                                      std=episode_stds[idx])
                    elif episode_intrinsic_rewards:
                        self.policy_record.add_result(np.average(episode_returns[idx]),
                                                      np.average(episode_red_blue_damages[idx]),
                                                      np.average(episode_red_red_damages[idx]),
                                                      np.average(episode_blue_red_damages[idx]),
                                                      episode_lengths[idx],
                                                      intrinsic_reward=episode_intrinsic_rewards[idx])
                    else:
                        self.policy_record.add_result(np.average(episode_returns[idx]),
                                                      np.average(episode_red_blue_damages[idx]),
                                                      np.average(episode_red_red_damages[idx]),
                                                      np.average(episode_blue_red_damages[idx]),
                                                      episode_lengths[idx])
                self.policy_record.save()


    def save_metrics(self, episode_returns, episode_lengths, episode_red_blue_damages, episode_red_red_damages,
                     episode_blue_red_damages, eval_mode=False, episode_stds=None):

        if len(episode_lengths) == 0: return

        if self.policy_record:
            if not eval_mode:
                for idx in range(len(episode_lengths)):
                    if episode_red_blue_damages[0] is None: continue
                    elif episode_stds:
                        self.policy_record.add_result(np.average(episode_returns[idx]),
                                                      np.average(episode_red_blue_damages[idx]),
                                                      np.average(episode_red_red_damages[idx]),
                                                      np.average(episode_blue_red_damages[idx]),
                                                      episode_lengths[idx],
                                                      std=episode_stds[idx])
                    else:
                        self.policy_record.add_result(np.average(episode_returns[idx]),
                                                      np.average(episode_red_blue_damages[idx]),
                                                      np.average(episode_red_red_damages[idx]),
                                                      np.average(episode_blue_red_damages[idx]),
                                                      episode_lengths[idx])
                self.policy_record.save()
