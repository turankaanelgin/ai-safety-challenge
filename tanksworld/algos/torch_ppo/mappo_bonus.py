import pdb
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F

import os
import json
import pickle
import cv2
import scipy
import matplotlib
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler

from tanksworld.minimap_util import *
from .heuristics import *
from . import core
from .mappo import PPOPolicy as MAPPOPolicy
from .mappo import RolloutBuffer


device = torch.device('cuda')


class PPOBonusPolicy(MAPPOPolicy):

    def __init__(self, env, callback, eval_mode=False, visual_mode=False, data_mode=False, **kargs):

        super(PPOBonusPolicy, self).__init__(env, callback, eval_mode, visual_mode, data_mode, **kargs)

    def setup_model(self, actor_critic, pi_lr, vf_lr, ac_kwargs, enemy_model=None):

        super().setup_model(actor_critic, pi_lr, vf_lr, ac_kwargs, enemy_model)

        self.replay_buffer = []
        self.replay_buffer_actions = []
        self.rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.phi_dim)
        self.rbf_feature.fit(X=np.random.randn(5, 9216 + 3))

    def compute_reward_bonus(self, states, actions):

        with torch.no_grad():
            states_reshaped = torch.flatten(states, end_dim=1).cuda()
            states = self.ac_model.pi.cnn_net(states_reshaped)
        phi = self.compute_kernel(states, actions)
        reward_bonus = torch.sqrt((torch.mm(phi, self.density_model) * phi).sum(1)).detach()
        return reward_bonus

    def compute_kernel(self, states, actions):

        np_states = states.cpu().numpy()
        np_actions = torch.flatten(actions, end_dim=1).cpu().numpy()
        states_acts_cat = np.concatenate((np_states, np_actions), axis=1)
        phi = self.rbf_feature.transform(states_acts_cat)
        phi = torch.tensor(phi).cuda()
        return phi

    def update_density_model(self):

        states = torch.cat(sum(self.replay_buffer, []))
        actions = torch.cat(sum(self.replay_buffer_actions, []))

        N = states.shape[0]
        ind = np.random.choice(N, min(2000, N), replace=False)
        with torch.no_grad():
            states_reshaped = torch.flatten(states, end_dim=1).cuda()
            states = self.ac_model.pi.cnn_net(states_reshaped)
        pdists = scipy.spatial.distance.pdist((states.cpu().numpy())[ind])
        self.rbf_feature.gamma = 1. / (np.median(pdists) ** 2)
        phi = self.compute_kernel(states, actions)
        n, d = phi.shape
        sigma = torch.mm(phi.t(), phi) + self.ridge * torch.eye(d).cuda()
        self.density_model = torch.inverse(sigma).detach()

    def gather_trajectories(self, batch_size):

        states = self.obs
        network = self.ac_model

        buffer = RolloutBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=batch_size)

        for _ in range(batch_size):
            actions, values, logp, _ = network.step(torch.as_tensor(states, dtype=torch.float32).to(device))
            next_states, rewards, terminals, info = self.env.step(actions.cpu().numpy())
            buffer.store(torch.as_tensor(states, dtype=torch.float32),
                         actions,
                         torch.as_tensor(rewards, dtype=torch.float32),
                         values.detach(),
                         logp.detach(),
                         terminals)
            states = next_states

        with torch.no_grad():
            _, values, _, _ = network.step(torch.as_tensor(states, dtype=torch.float32).cuda())
        buffer.finish_path(values, env_idx=0)
        return buffer.get()

    def update_replay_buffer(self, batch_size):

        states, actions, returns, infos = [], [], [], []
        for _ in range(self.n_rollouts_for_density_est):
            data = self.gather_trajectories(batch_size=batch_size)
            states += data['obs']
            returns += data['ret']
            actions += data['act']

        self.replay_buffer.append(states)
        self.replay_buffer_actions.append(actions)
        if len(self.replay_buffer) > 10:
            self.replay_buffer.pop(0)
            self.replay_buffer_actions.pop(0)


    def learn(self, actor_critic=core.ActorCritic, ac_kwargs=dict(), seed=-1,
              steps_per_epoch=800, steps_to_run=100000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
              vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97,
              target_kl=0.01, freeze_rep=True, entropy_coef=0.0, use_value_norm=False,
              tb_writer=None, selfplay=False, ally_heuristic=False, enemy_heuristic=False, dense_reward=False,
              centralized=False, centralized_critic=False, local_std=False, enemy_model=None, single_agent=False,
              discrete_action=False, rnd=False, rnd_bonus=0.0, ridge=0.01, phi_dim=10, **kargs):

        env = self.env
        self.writer = tb_writer
        self.loss_p_index, self.loss_v_index = 0, 0
        self.set_random_seed(seed)

        ac_kwargs['init_log_std'] = kargs['init_log_std']
        ac_kwargs['centralized'] = centralized
        ac_kwargs['centralized_critic'] = centralized_critic
        ac_kwargs['local_std'] = local_std
        ac_kwargs['discrete_action'] = discrete_action

        self.centralized = centralized
        self.centralized_critic = centralized_critic
        self.selfplay = selfplay
        self.single_agent = single_agent
        self.discrete_action = discrete_action
        self.rnd = rnd
        self.rnd_bonus = rnd_bonus
        self.ridge = ridge
        self.phi_dim = phi_dim
        self.n_rollouts_for_density_est = 1

        print('POLICY SEED', seed)

        self.prev_ckpt = None
        self.setup_model(actor_critic, pi_lr, vf_lr, ac_kwargs, enemy_model=enemy_model)
        self.load_model(kargs['model_path'], kargs['cnn_model_path'], freeze_rep, steps_per_epoch)
        num_envs = kargs['n_envs']
        if self.callback:
            self.callback.init_model(self.ac_model)

        ep_ret = 0
        ep_ret_intrinsic = 0
        ep_len = 0
        ep_rb_dmg = np.zeros(num_envs)
        ep_br_dmg = np.zeros(num_envs)
        ep_rr_dmg = np.zeros(num_envs)

        buf = RolloutBuffer(self.obs_dim, self.act_dim, steps_per_epoch, gamma,
                            lam, n_rollout_threads=num_envs, centralized=centralized,
                            n_agents=1 if single_agent else 5,
                            discrete_action=discrete_action)

        if not os.path.exists(kargs['save_dir']):
            from pathlib import Path
            Path(kargs['save_dir']).mkdir(parents=True, exist_ok=True)

        step = self.start_step
        episode_lengths = []
        episode_returns = []
        episode_intrinsic_returns = []
        episode_red_blue_damages, episode_red_red_damages, episode_blue_red_damages = [], [], []
        episode_stds = []
        # Damage for last hundred steps
        last_hundred_red_blue_damages = [[] for _ in range(num_envs)]
        last_hundred_red_red_damages = [[] for _ in range(num_envs)]
        last_hundred_blue_red_damages = [[] for _ in range(num_envs)]
        best_eval_score = self.best_eval_score

        if ally_heuristic or enemy_heuristic or dense_reward:
            mixing_coeff = 0.6

        self.update_replay_buffer(batch_size=steps_per_epoch)
        self.update_density_model()

        while step < steps_to_run:

            if (step + 1) % 50000 == 0 or step == 0:  # Periodically save the model
                self.save_model(kargs['save_dir'], step)

            if (step + 1) % 100000 == 0 and selfplay:  # Selfplay load enemy
                if self.prev_ckpt is not None:
                    self.enemy_model.load_state_dict(self.prev_ckpt)

            if (step + 1) % 50000 == 0:  # Selfplay record prev checkpoint
                self.prev_ckpt = self.ac_model.state_dict()

            if (step + 1) % 25000 == 0 and (ally_heuristic or enemy_heuristic):  # Heuristic anneal mixing coefficient
                if mixing_coeff >= 0.05:
                    mixing_coeff -= 0.05

            if (step + 1) % 10000 == 0 and dense_reward:
                if mixing_coeff >= 0.1: mixing_coeff -= 0.1

            step += 1

            obs = torch.as_tensor(self.obs, dtype=torch.float32).to(device)

            if selfplay or enemy_model is not None:
                ally_obs = obs[:, :5, :, :, :]
                ally_a, v, logp, entropy = self.ac_model.step(ally_obs)
                enemy_obs = obs[:, 5:, :, :, :]
                with torch.no_grad():
                    enemy_a, _, _, _ = self.enemy_model.step(enemy_obs)
                a = torch.cat((ally_a, enemy_a), dim=1)

            else:
                a, v, logp, entropy = self.ac_model.step(obs)

            if ally_heuristic or enemy_heuristic:
                if ally_heuristic:
                    heuristic_action = get_ally_heuristic_2(self.state_vector)
                else:
                    heuristic_action = get_enemy_heuristic(self.state_vector)

                # Mix heuristic action with policy action

                coin = np.random.rand()
                if coin < mixing_coeff:
                    next_obs, r, terminal, info = env.step(heuristic_action)
                else:
                    next_obs, r, terminal, info = env.step(a.cpu().numpy())

            else:
                if discrete_action:
                    action1 = (a // 100) * 0.04 - 1
                    action2 = ((a % 100) // 2) * 0.04 - 1
                    action3 = (a % 100) % 2 - 0.5
                    action = torch.cat((action1.unsqueeze(-1), action2.unsqueeze(-1), action3.unsqueeze(-1)), dim=-1)
                    next_obs, r, terminal, info = env.step(action.cpu().numpy())
                else:
                    next_obs, r, terminal, info = env.step(a.cpu().numpy())

            if self.rnd:
                rnd_target = self.rnd_network(obs).detach()
                rnd_pred = self.rnd_pred_network(obs).detach()
                rnd_loss = F.mse_loss(rnd_pred, rnd_target, reduction='none').mean(2)
                r += rnd_loss.detach().cpu().numpy()

            bonus = self.compute_reward_bonus(obs, a).unsqueeze(0).cpu().numpy()

            ep_ret += np.average(np.sum(r, axis=1))
            ep_ret_intrinsic += np.average(np.sum(bonus, axis=1))
            ep_len += 1

            r += bonus

            self.state_vector = [info[env_idx]['state_vector'] for env_idx in range(len(info))]

            if selfplay or enemy_model is not None:
                r = r[:, :5]
                a = ally_a
                obs = ally_obs

            if dense_reward:
                distances = distance_to_closest_enemy(self.state_vector, obs,
                                                      num_agents=1 if single_agent else 5)
                distances = 0.001 * np.asarray(distances)
                r = r - mixing_coeff * np.expand_dims(distances, axis=0)

            r = torch.as_tensor(r, dtype=torch.float32).to(device)
            self.obs = next_obs

            buf.store(obs, a, r, v, logp, terminal)

            for env_idx, done in enumerate(terminal):
                if done:
                    stats = info[env_idx]['red_stats']
                    ep_rr_dmg[env_idx] = stats['damage_inflicted_on']['ally']
                    ep_rb_dmg[env_idx] = stats['damage_inflicted_on']['enemy']
                    ep_br_dmg[env_idx] = stats['damage_taken_by']['enemy']
                    last_hundred_red_blue_damages[env_idx].append(ep_rb_dmg[env_idx])
                    last_hundred_red_red_damages[env_idx].append(ep_rr_dmg[env_idx])
                    last_hundred_blue_red_damages[env_idx].append(ep_br_dmg[env_idx])
                    last_hundred_red_blue_damages[env_idx] = last_hundred_red_blue_damages[env_idx][-100:]
                    last_hundred_red_red_damages[env_idx] = last_hundred_red_red_damages[env_idx][-100:]
                    last_hundred_blue_red_damages[env_idx] = last_hundred_blue_red_damages[env_idx][-100:]

            epoch_ended = step > 0 and step % steps_per_epoch == 0

            if np.any(terminal) or epoch_ended:

                with torch.no_grad():
                    obs_input = self.obs[:, :5, :, :, :] if selfplay or enemy_model is not None else self.obs
                    _, v, _, _ = self.ac_model.step(
                        torch.as_tensor(obs_input, dtype=torch.float32).to(device))

                for env_idx, done in enumerate(terminal):
                    if done:
                        with torch.no_grad(): v[env_idx] = 0 if self.centralized or single_agent else torch.zeros(5)
                    buf.finish_path(v, env_idx)

                if epoch_ended:
                    for env_idx in range(num_envs):
                        buf.finish_path(v, env_idx)

                episode_lengths.append(ep_len)
                episode_returns.append(ep_ret)
                episode_intrinsic_returns.append(ep_ret_intrinsic)
                episode_red_red_damages.append(ep_rr_dmg)
                episode_blue_red_damages.append(ep_br_dmg)
                episode_red_blue_damages.append(ep_rb_dmg)
                std = torch.exp(
                    self.ac_model.pi.log_std).cpu().detach().numpy() if not discrete_action else torch.zeros((3))
                episode_stds.append(std)

                if epoch_ended:
                    self.update(buf, train_pi_iters, train_v_iters, target_kl, clip_ratio, entropy_coef)

                ep_ret = 0
                ep_ret_intrinsic = 0
                ep_len = 0
                ep_rb_dmg = np.zeros(num_envs)
                ep_br_dmg = np.zeros(num_envs)
                ep_rr_dmg = np.zeros(num_envs)

            if (step + 1) % 500 == 0:

                self.update_replay_buffer(batch_size=steps_per_epoch)
                self.update_density_model()

            if (step + 1) % 100 == 0:

                if self.callback:
                    self.callback.save_metrics_multienv(episode_returns, episode_lengths, episode_red_blue_damages,
                                                        episode_red_red_damages, episode_blue_red_damages,
                                                        episode_intrinsic_rewards=episode_intrinsic_returns)

                    with open(os.path.join(self.callback.policy_record.data_dir, 'mean_statistics.json'), 'w+') as f:
                        if last_hundred_red_blue_damages[0] is not None:
                            red_red_damage = np.average(np.concatenate(last_hundred_red_red_damages))
                            red_blue_damage = np.average(np.concatenate(last_hundred_red_blue_damages))
                            blue_red_damage = np.average(np.concatenate(last_hundred_blue_red_damages))
                        else:
                            red_red_damage, red_blue_damage, blue_red_damage = 0.0, 0.0, 0.0

                        json.dump({'Red-Blue-Damage': red_blue_damage,
                                   'Red-Red-Damage': red_red_damage,
                                   'Blue-Red-Damage': blue_red_damage}, f, indent=True)

                episode_lengths = []
                episode_returns = []
                episode_red_blue_damages = []
                episode_blue_red_damages = []
                episode_red_red_damages = []
                episode_stds = []

            if (step + 1) % 50000 == 0:

                if self.callback and self.callback.val_env:

                    eval_score = self.callback.validate_policy(self.ac_model.state_dict(), device, discrete_action)
                    if eval_score > best_eval_score:
                        self.save_model(kargs['save_dir'], step, is_best=True)
                        best_eval_score = eval_score
                        with open(os.path.join(self.callback.policy_record.data_dir, 'best_eval_score.json'),
                                  'w+') as f:
                            json.dump(best_eval_score, f)

                        if self.callback.eval_env:
                            self.callback.evaluate_policy(self.ac_model.state_dict(), device, step)
