import pdb
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector as flatten
from torch.nn.utils import vector_to_parameters as assign
from .torch_utils import *

import os
import json
import pickle
import cv2
import matplotlib
import matplotlib.pyplot as plt

from tanksworld.minimap_util import *
from . import core


device = torch.device('cuda')


class RolloutBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95,
                 n_rollout_threads=1, centralized=False, n_agents=5, discrete_action=False):

        self.n_agents = n_agents
        self.obs_buf = torch.zeros(core.combined_shape_v3(size, n_rollout_threads, self.n_agents, obs_dim)).to(device)
        if discrete_action:
            self.act_buf = torch.zeros((size, n_rollout_threads, self.n_agents)).to(device)
        else:
            self.act_buf = torch.zeros(core.combined_shape_v3(size, n_rollout_threads, self.n_agents, act_dim)).to(device)
        self.adv_buf = torch.zeros((size, n_rollout_threads, self.n_agents)).to(device)
        self.rew_buf = torch.zeros((size, n_rollout_threads, self.n_agents)).to(device)
        self.ret_buf = torch.zeros((size, n_rollout_threads, self.n_agents)).to(device)
        self.val_buf = torch.zeros((size, n_rollout_threads, self.n_agents)).to(device)
        self.logp_buf = torch.zeros((size, n_rollout_threads, self.n_agents)).to(device)
        self.episode_starts = torch.zeros((size, n_rollout_threads, self.n_agents)).to(device)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size
        self.path_start_idx = np.zeros(n_rollout_threads,)
        self.n_rollout_threads = n_rollout_threads
        self.buffer_size = size
        self.centralized = centralized

    def store(self, obs, act, rew, val, logp, dones):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """

        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs.squeeze(2)
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.episode_starts[self.ptr] = torch.FloatTensor(dones).unsqueeze(1).tile((1, self.n_agents))
        self.ptr += 1

    def finish_path(self, last_val, env_idx):

        path_start = int(self.path_start_idx[env_idx])

        last_val = last_val[env_idx,:].unsqueeze(0)
        rews = torch.cat((self.rew_buf[path_start:self.ptr, env_idx], last_val), dim=0)
        vals = torch.cat((self.val_buf[path_start:self.ptr, env_idx], last_val), dim=0)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        discount_delta = core.discount_cumsum(deltas.cpu().numpy(), self.gamma * self.lam)
        self.adv_buf[path_start:self.ptr, env_idx] = torch.as_tensor(discount_delta.copy(), dtype=torch.float32).to(device)

        discount_rews = core.discount_cumsum(rews.cpu().numpy(), self.gamma)[:-1]
        self.ret_buf[path_start:self.ptr, env_idx] = torch.as_tensor(discount_rews.copy(), dtype=torch.float32).to(
            device)

        self.path_start_idx[env_idx] = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        # self.ptr, self.path_start_idx = 0, 0
        self.ptr = 0
        self.path_start_idx = np.zeros(self.n_rollout_threads, )
        # the next two lines implement the advantage normalization trick
        adv_buf = self.adv_buf.flatten(start_dim=1)
        adv_std, adv_mean = torch.std_mean(adv_buf, dim=0)
        adv_buf = (adv_buf - adv_mean) / adv_std
        self.adv_buf = adv_buf.reshape(adv_buf.shape[0], self.n_rollout_threads, self.n_agents)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in data.items()}


class TRPOPolicy():

    def __init__(self, env, callback, eval_mode=False, **kargs):
        self.kargs = kargs
        self.env = env
        self.callback = callback
        self.eval_mode = eval_mode

    def run(self, num_steps):

        self.kargs.update({'steps_to_run': num_steps})

        if self.eval_mode:
            self.evaluate(episodes_to_run=num_steps, model_path=self.kargs['model_path'],
                          num_envs=self.kargs['n_envs'], ac_kwargs=ac_kwargs)
        else:
            self.learn(**self.kargs)

    def set_random_seed(self, seed):

        # Random seed
        if seed == -1:
            MAX_INT = 2147483647
            seed = np.random.randint(MAX_INT)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def setup_model(self, actor_critic, pi_lr, vf_lr, ac_kwargs):

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.obs = self.env.reset()
        self.state_vector = np.zeros((12, 6))

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
        self.pi_optimizer = Adam(self.ac_model.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac_model.v.parameters(), lr=vf_lr)

    def load_model(self, model_path, cnn_model_path, freeze_rep, steps_per_epoch):

        self.start_step = 0
        self.best_eval_score = -np.infty
        # Load from previous checkpoint
        if model_path:
            ckpt = torch.load(model_path)
            self.ac_model.load_state_dict(ckpt['model_state_dict'], strict=True)
            self.pi_optimizer.load_state_dict(ckpt['pi_optimizer_state_dict'])
            self.vf_optimizer.load_state_dict(ckpt['vf_optimizer_state_dict'])
            self.start_step = ckpt['step']
            self.start_step -= self.start_step % steps_per_epoch
            if os.path.exists(os.path.join(self.callback.policy_record.data_dir, 'best_eval_score.json')):
                with open(os.path.join(self.callback.policy_record.data_dir, 'best_eval_score.json'), 'r') as f:
                    self.best_eval_score = json.load(f)

            if self.selfplay:
                self.enemy_model.load_state_dict(ckpt['enemy_model_state_dict'], strict=True)
                self.prev_ckpt = ckpt['model_state_dict']

        # Only load the representation part
        elif cnn_model_path:
            state_dict = torch.load(cnn_model_path)

            temp_state_dict = {}
            for key in state_dict:
                if 'cnn_net' in key:
                    temp_state_dict[key] = state_dict[key]

            self.ac_model.load_state_dict(temp_state_dict, strict=False)

        if freeze_rep:
            for name, param in self.ac_model.named_parameters():
                if 'cnn_net' in name:
                    param.requires_grad = False

        from torchinfo import summary
        summary(self.ac_model)

    def save_model(self, save_dir, step, is_best=False):

        if is_best:
            model_path = os.path.join(save_dir, 'best.pth')
        else:
            model_path = os.path.join(save_dir, str(step) + '.pth')
        ckpt_dict = {'step': step,
                     'model_state_dict': self.ac_model.state_dict(),
                     'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),
                     'vf_optimizer_state_dict': self.vf_optimizer.state_dict()}
        torch.save(ckpt_dict, model_path)

    def surrogate_reward(self, adv, *, new, old):

        log_ps_new, log_ps_old = new, old
        n_advs = adv

        assert shape_equal_cmp(log_ps_new, log_ps_old, n_advs)

        ratio_new_old = torch.exp(log_ps_new - log_ps_old)
        return ratio_new_old * n_advs

    def get_policy_parameters(self):

        for param in self.ac_model.pi.parameters():
            if param.requires_grad:
                yield param

    # Set up function for computing TRPO policy loss
    def compute_loss_pi(self, data, fisher_frac_samples, damping, cg_steps, max_kl, max_backtrack):

        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        all_states = torch.flatten(obs, end_dim=1)
        actions = torch.flatten(act, end_dim=1)
        advs = torch.flatten(adv, end_dim=1)
        old_log_ps = torch.flatten(logp_old, end_dim=1)

        initial_parameters = flatten(self.get_policy_parameters()).clone()
        _, action_log_probs = self.ac_model.pi(all_states, actions)
        pds = self.ac_model.pi._mean_std(all_states)

        surr_rew = self.surrogate_reward(advs, new=action_log_probs, old=old_log_ps).mean()
        grad = torch.autograd.grad(surr_rew, self.get_policy_parameters(), retain_graph=True)
        flat_grad = flatten(grad)

        num_samples = int(all_states.shape[0] * fisher_frac_samples)
        selected = np.random.choice(range(all_states.shape[0]), num_samples, replace=False)

        detached_selected_pds = select_prob_dists(pds, selected, detach=True)
        selected_pds = select_prob_dists(pds, selected, detach=False)

        kl = self.ac_model.pi.calc_kl(detached_selected_pds, selected_pds, get_mean=True)
        g = flatten(torch.autograd.grad(kl, self.get_policy_parameters(), create_graph=True))

        if torch.any(torch.isnan(g)): pdb.set_trace()

        def fisher_product(x, damp_coef=1.):
            contig_flat = lambda q: torch.cat([y.contiguous().view(-1) for y in q])
            z = g @ x
            hv = torch.autograd.grad(z, self.get_policy_parameters(), retain_graph=True)
            return contig_flat(hv).detach() + x * damping * damp_coef

        step = cg_solve(fisher_product, flat_grad, cg_steps)

        if torch.any(torch.isnan(step)): pdb.set_trace()

        max_step_coeff = (2 * max_kl / (step @ fisher_product(step)))**(0.5)
        max_trpo_step = max_step_coeff * step

        with torch.no_grad():
            def backtrack_fn(s):
                assign(initial_parameters + s.data, self.get_policy_parameters())
                test_pds = self.ac_model.pi._mean_std(all_states)
                _, test_action_log_probs = self.ac_model.pi(all_states, actions)
                new_reward = self.surrogate_reward(advs, new=test_action_log_probs, old=old_log_ps).mean()
                if new_reward <= surr_rew or self.ac_model.pi.calc_kl(pds, test_pds, get_mean=True) > max_kl:
                    return -float('inf')
                return new_reward - surr_rew

            expected_improve = flat_grad @ max_trpo_step
            final_step = backtracking_line_search(backtrack_fn, max_trpo_step,
                                                  expected_improve,
                                                  num_tries=max_backtrack)
            if torch.any(torch.isnan(final_step)): pdb.set_trace()

            assign(initial_parameters + final_step, self.get_policy_parameters())

        return surr_rew

    # Set up function for computing value loss
    def compute_loss_v(self, data):

        obs, ret = data['obs'], data['ret']
        obs = torch.flatten(obs, end_dim=2)
        ret = torch.flatten(ret)
        values = self.ac_model.v(obs).squeeze(0)
        return ((values - ret) ** 2).mean()


    def update(self, buf, train_pi_iters, train_v_iters, fisher_frac_samples, damping, cg_steps, max_kl, max_backtrack):

        data = buf.get()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data, fisher_frac_samples, damping, cg_steps, max_kl, max_backtrack)

            #kl = pi_info['kl']
            #if kl > 1.5 * target_kl:
            #    break

            loss_pi.backward()

            self.loss_p_index += 1
            self.writer.add_scalar('loss/Policy_Loss', loss_pi, self.loss_p_index)
            std = torch.exp(self.ac_model.pi.log_std)
            self.writer.add_scalar('std_dev/move', std[0].item(), self.loss_p_index)
            self.writer.add_scalar('std_dev/turn', std[1].item(), self.loss_p_index)
            self.writer.add_scalar('std_dev/shoot', std[2].item(), self.loss_p_index)
            self.pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.loss_v_index += 1
            self.writer.add_scalar('loss/Value_Loss', loss_v, self.loss_v_index)
            self.vf_optimizer.step()


    def learn(self, actor_critic=core.ActorCritic, ac_kwargs=dict(), seed=-1,
              steps_per_epoch=800, steps_to_run=100000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
              vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97,
              freeze_rep=True,
              tb_writer=None, selfplay=False, ally_heuristic=False, enemy_heuristic=False, dense_reward=False,
              centralized=False, enemy_model=None, single_agent=False,
              discrete_action=False, rnd=False, noisy=False, **kargs):

        env = self.env
        self.writer = tb_writer
        self.loss_p_index, self.loss_v_index = 0, 0
        self.set_random_seed(seed)

        print('POLICY SEED', seed)

        self.prev_ckpt = None
        self.setup_model(actor_critic, pi_lr, vf_lr, ac_kwargs)
        self.load_model(kargs['model_path'], kargs['cnn_model_path'], freeze_rep, steps_per_epoch)
        num_envs = kargs['n_envs']
        if self.callback:
            self.callback.init_model(self.ac_model)

        ep_ret = 0
        ep_intrinsic_ret = 0
        ep_len = 0
        ep_rb_dmg = np.zeros(num_envs)
        ep_br_dmg = np.zeros(num_envs)
        ep_rr_dmg = np.zeros(num_envs)

        buf = RolloutBuffer(self.obs_dim, self.act_dim, steps_per_epoch, gamma,
                            lam, n_rollout_threads=num_envs, centralized=centralized,
                            n_agents=5,
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

        while step < steps_to_run:

            if noisy: self.ac_model.resample()

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

            a, v, logp, entropy = self.ac_model.step(obs)
            next_obs, r, terminal, info = env.step(a.cpu().numpy())
            extrinsic_reward = r.copy()

            self.state_vector = [info[env_idx]['state_vector'] for env_idx in range(len(info))]

            ep_ret += np.average(np.sum(extrinsic_reward, axis=1))
            ep_len += 1

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
                    obs_input = self.obs
                    _, v, _, _ = self.ac_model.step(
                        torch.as_tensor(obs_input, dtype=torch.float32).to(device))

                for env_idx, done in enumerate(terminal):
                    if done:
                        with torch.no_grad(): v[env_idx] = torch.zeros(5)
                    buf.finish_path(v, env_idx)

                if epoch_ended:
                    for env_idx in range(num_envs):
                        buf.finish_path(v, env_idx)

                episode_lengths.append(ep_len)
                episode_returns.append(ep_ret)
                episode_red_red_damages.append(ep_rr_dmg)
                episode_blue_red_damages.append(ep_br_dmg)
                episode_red_blue_damages.append(ep_rb_dmg)
                std = torch.exp(
                    self.ac_model.pi.log_std).cpu().detach().numpy() if not discrete_action else torch.zeros((3))
                episode_stds.append(std)

                if epoch_ended:
                    self.update(buf, train_pi_iters, train_v_iters, kargs['fisher_frac_samples'], kargs['damping'],
                                kargs['cg_steps'], kargs['max_kl'], kargs['max_backtrack'])

                ep_ret = 0
                ep_intrinsic_ret = 0
                ep_len = 0
                ep_rb_dmg = np.zeros(num_envs)
                ep_br_dmg = np.zeros(num_envs)
                ep_rr_dmg = np.zeros(num_envs)

            if (step + 1) % 100 == 0:

                if self.callback:
                    self.callback.save_metrics_multienv(episode_returns, episode_lengths, episode_red_blue_damages,
                                                        episode_red_red_damages, episode_blue_red_damages,
                                                        episode_stds=episode_stds if not rnd else None,
                                                        episode_intrinsic_rewards=episode_intrinsic_returns if rnd else None)

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
                        self.callback.evaluate_policy(self.ac_model.state_dict(), device)
