import json, os
import pdb

import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue

from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper


def evaluate_policy(eval_env, model, eval_steps):

    steps = 0
    observation = eval_env.reset()
    model.eval()

    eplen, epret = 0, 0
    ep_rr_damage, ep_rb_damage, ep_br_damage = 0, 0, 0
    episode_returns, episode_lengths = [], []
    episode_red_blue_damages, episode_blue_red_damages = [], []
    episode_red_red_damages = []

    while steps < eval_steps:

        with torch.no_grad():
            action, v, logp, _ = model(torch.as_tensor(observation, dtype=torch.float32).to(device))

        observation, reward, done, info = eval_env.step(action)

        steps += 1
        eplen += 1
        epret += np.sum(reward)
        stats = info[0]['current']
        ep_rr_damage += stats['red_ally_damage']
        ep_rb_damage += stats['red_enemy_damage']
        ep_br_damage += stats['blue_enemy_damage']

        if done[0]:
            episode_returns.append(epret)
            episode_lengths.append(eplen)
            episode_red_red_damages.append(ep_rr_damage)
            episode_blue_red_damages.append(ep_br_damage)
            episode_red_blue_damages.append(ep_rb_damage)
            eplen, epret = 0, 0
            ep_rr_damage, ep_rb_damage, ep_br_damage = 0, 0, 0
            steps += 1

        # if steps % 50 == 0:
        #    self.save_metrics(info, episode_returns, episode_lengths, episode_red_blue_damages,
        #                      episode_red_red_damages, episode_blue_red_damages, eval_mode=True)


class EvalCallback:

    def __init__(self, env, policy_record, eval_steps=10, eval_env=None):
        self.env = env
        self.model = None
        self.policy_record = policy_record
        self.eval_env = eval_env
        self.eval_steps = eval_steps
        self.n_calls = 0
        self.eval_freq = 1000
        self.last_mean_reward = 0.0
        self.last_mean_damage_inflicted = 0.0
        self.last_mean_damage_taken = 0.0

    def init_model(self, model):
        self.model = model

    def evaluate_policy_modified(self, model_state_dict, device):

        steps = 0
        observation = self.eval_env.reset()

        self.model.load_state_dict(model_state_dict, strict=True)
        self.model.eval()

        episode_red_blue_damages, episode_blue_red_damages = [], []
        episode_red_red_damages = []

        while steps < self.eval_steps:
            with torch.no_grad():
                action, _, _, _ = self.model.step(torch.as_tensor(observation, dtype=torch.float32).to(device))
            observation, reward, done, info = self.eval_env.step(action.cpu().numpy())

            if done[0]:
                ep_rr_damage = info[0]['red_stats']['damage_inflicted_on']['ally']
                ep_rb_damage = info[0]['red_stats']['damage_inflicted_on']['enemy']
                ep_br_damage = info[0]['red_stats']['damage_taken_by']['enemy']

                episode_red_red_damages.append(ep_rr_damage)
                episode_blue_red_damages.append(ep_br_damage)
                episode_red_blue_damages.append(ep_rb_damage)

                steps += 1
                observation = self.eval_env.reset()

                if steps == self.eval_steps:
                    avg_red_red_damages = np.mean(episode_red_red_damages)
                    avg_red_blue_damages = np.mean(episode_red_blue_damages)
                    avg_blue_red_damages = np.mean(episode_blue_red_damages)

                    with open(os.path.join(self.policy_record.data_dir, 'mean_eval_statistics.json'), 'w+') as f:
                        json.dump({'Number of games': steps,
                                   'Red-Red-Damage': avg_red_red_damages.tolist(),
                                   'Red-Blue Damage': avg_red_blue_damages.tolist(),
                                   'Blue-Red Damage': avg_blue_red_damages.tolist()}, f, indent=4)

    def evaluate_policy(self):

        steps = 0
        observation = self.eval_env.reset()
        model = torch.load(self.model_state_dict)

        eplen, epret = 0, 0
        ep_rr_damage, ep_rb_damage, ep_br_damage = 0, 0, 0
        episode_returns, episode_lengths = [], []
        episode_red_blue_damages, episode_blue_red_damages = [], []
        episode_red_red_damages = []

        while steps < self.eval_steps:

            with torch.no_grad():
                action, v, logp, _ = self.model.step(torch.as_tensor(observation, dtype=torch.float32))

            observation, reward, done, info = self.eval_env.step(action)

            steps += 1
            eplen += 1
            epret += np.sum(reward)
            stats = info[0]['current']
            ep_rr_damage += stats['red_ally_damage']
            ep_rb_damage += stats['red_enemy_damage']
            ep_br_damage += stats['blue_enemy_damage']

            if done[0]:
                episode_returns.append(epret)
                episode_lengths.append(eplen)
                episode_red_red_damages.append(ep_rr_damage)
                episode_blue_red_damages.append(ep_br_damage)
                episode_red_blue_damages.append(ep_rb_damage)
                eplen, epret = 0, 0
                ep_rr_damage, ep_rb_damage, ep_br_damage = 0, 0, 0
                steps += 1

            if steps % 50 == 0:
                self.save_metrics(info, episode_returns, episode_lengths, episode_red_blue_damages,
                                  episode_red_red_damages, episode_blue_red_damages, eval_mode=True)


    def _on_step(self):

        self.n_calls += 1
        episode_rewards = self.policy_record.channels['main'].ep_results
        episode_red_red_damage = self.policy_record.channels['main'].ep_red_red_damages
        episode_red_blue_damage = self.policy_record.channels['main'].ep_red_blue_damages
        episode_blue_red_damage = self.policy_record.channels['main'].ep_blue_red_damages
        if len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards[-100:])
            mean_damage_inflicted = np.mean(episode_red_blue_damage[-100:])
            mean_damage_taken = np.mean(episode_blue_red_damage[-100:])+np.mean(episode_red_red_damage[-100:])
            self.last_mean_reward = mean_reward
            self.last_mean_damage_inflicted = mean_damage_inflicted
            self.last_mean_damage_taken = mean_damage_taken
        '''
        if self.n_calls % self.eval_freq == 0 or self.n_calls == 1:
            
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
            ctx = mp.get_context(start_method)
            remote, work_remote = ctx.Pipe()
            process = ctx.Process(target=evaluate_policy, args=(work_remote, remote, self.eval_env, self.model), daemon=True)
            process.start()
            process.join()
            

            self.model_state_dict = self.model.state_dict()
            self.model_type
            p = Process(target=self.evaluate_policy, args=())
            p.start()
            p.join()
        '''

    def save_metrics_modified(self, episode_returns, episode_lengths, episode_red_blue_damages, episode_red_red_damages,
                                episode_blue_red_damages, eval_mode=False):

        assert len(episode_lengths) == len(episode_returns) == len(episode_red_red_damages) == len(episode_red_blue_damages) \
               == len(episode_blue_red_damages)

        if len(episode_lengths) == 0: return

        episode_stats = {'Red-Blue-Damage': np.mean(episode_red_blue_damages),
                         'Red-Red-Damage': np.mean(episode_red_red_damages),
                         'Blue-Red-Damage': np.mean(episode_blue_red_damages)}
        if self.policy_record:
            if not eval_mode:
                with open(os.path.join(self.policy_record.data_dir, 'mean_statistics.json'), 'w+') as f:
                    json.dump(episode_stats, f, indent=True)

                for idx in range(len(episode_lengths)):
                    self.policy_record.add_result(episode_returns[idx], episode_red_blue_damages[idx],
                                                  episode_red_red_damages[idx], episode_blue_red_damages[idx],
                                                  episode_lengths[idx])
                self.policy_record.save()


    def save_metrics(self, info, episode_returns, episode_lengths, episode_red_blue_damages, episode_red_red_damages,
                     episode_blue_red_damages, eval_mode=False):

        assert len(episode_lengths) == len(episode_returns) == len(episode_red_red_damages) == len(episode_red_blue_damages) \
                == len(episode_blue_red_damages)

        length = len(info[0]['all'])
        if length < 3: return

        episode_stats = info[0]['all'][-min(100, length):]
        episode_stats = {key: np.average([episode_stats[i][key] \
                                          for i in range(len(episode_stats))]) \
                                          for key in episode_stats[0]}
        if self.policy_record:
            if not eval_mode:
                with open(os.path.join(self.policy_record.data_dir, 'mean_statistics.json'), 'w+') as f:
                    json.dump(episode_stats, f, indent=True)

                for idx in range(len(episode_lengths)):
                    self.policy_record.add_result(episode_returns[idx], episode_red_blue_damages[idx],
                                                  episode_red_red_damages[idx], episode_blue_red_damages[idx],
                                                  episode_lengths[idx])
                self.policy_record.save()
            else:
                with open(os.path.join(self.policy_record.data_dir, 'mean_eval_statistics.json'), 'w+') as f:
                    json.dump(episode_stats, f, indent=True)


class TrialEvalCallback(EvalCallback):

    def __init__(self, trial, env, policy_record, eval_env=None):
        super(TrialEvalCallback, self).__init__(
            env, policy_record, eval_env
        )
        self.trial = trial
        self.is_pruned = False
        self.eval_idx = 0

    def _on_step(self):
        super(TrialEvalCallback, self)._on_step()
        self.eval_idx += 1
        self.trial.report(self.last_mean_reward, self.eval_idx)
        if self.trial.should_prune():
            self.is_pruned = True
            return False
        return True