import pdb
import numpy as np
import random
import torch
from torch.optim import Adam

import os
import json
import math
import pickle
import cv2
import matplotlib
import matplotlib.pyplot as plt

from tanksworld.minimap_util import *
from .heuristics import *
from . import core_ind as core


device = torch.device('cuda')


class RolloutBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):

        self.n_agents = 5
        self.obs_buf = torch.zeros(core.combined_shape_v2(size, self.n_agents, obs_dim)).to(device)
        self.act_buf = torch.zeros(core.combined_shape_v2(size, self.n_agents, act_dim)).to(device)
        self.adv_buf = torch.zeros((size, self.n_agents)).to(device)
        self.rew_buf = torch.zeros((size, self.n_agents)).to(device)
        self.ret_buf = torch.zeros((size, self.n_agents)).to(device)
        self.val_buf = torch.zeros((size, self.n_agents)).to(device)
        self.logp_buf = torch.zeros((size, self.n_agents)).to(device)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
                Append one timestep of agent-environment interaction to the buffer.
                """

        assert self.ptr < self.max_size  # buffer has to have room so you can store

        self.obs_buf[self.ptr] = obs.squeeze(2)
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)

        vals = torch.cat((self.val_buf[path_slice], last_val), dim=0)
        rews = torch.cat((self.rew_buf[path_slice], last_val), dim=0)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        discount_delta = core.discount_cumsum(deltas.cpu().numpy(), self.gamma * self.lam)
        self.adv_buf[path_slice] = torch.as_tensor(discount_delta.copy(), dtype=torch.float32).to(device)

        discount_rews = core.discount_cumsum(rews.cpu().numpy(), self.gamma)[:-1]
        self.ret_buf[path_slice] = torch.as_tensor(discount_rews.copy(), dtype=torch.float32).to(device)

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_std, adv_mean = torch.std_mean(self.adv_buf, dim=0)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in data.items()}



class PPOPolicy():

    def __init__(self, env, callback, eval_mode=False, **kargs):
        self.kargs = kargs
        self.env = env
        self.callback = callback
        self.eval_mode = eval_mode

    def run(self, num_steps):
        self.kargs.update({'steps_to_run': num_steps})

        ac_kwargs = {}
        ac_kwargs['init_log_std'] = self.kargs['init_log_std']
        ac_kwargs['centralized'] = self.kargs['centralized']
        ac_kwargs['centralized_critic'] = self.kargs['centralized_critic']
        ac_kwargs['local_std'] = self.kargs['local_std']

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


    def setup_model(self, actor_critic, pi_lr, vf_lr, ac_kwargs):

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.obs = self.env.reset()
        self.state_vector = np.zeros((5, 6))

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
        pi_parameters = []
        for agent_idx in range(5):
            pi_parameters += list(self.ac_model.pi[agent_idx].parameters())
        self.pi_optimizer = Adam(pi_parameters, lr=pi_lr)
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


    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data, clip_ratio_1, clip_ratio_2):

        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        num_agents = 5
        logp = []
        for agent_idx in range(num_agents):
            pi, logp_ = self.ac_model.pi[agent_idx](obs[:,agent_idx], act[:,agent_idx])
            logp.append(logp_)
        logp = torch.transpose(torch.cat(logp, dim=0), 1, 0)

        imp_weights = torch.exp(logp - logp_old)
        ratio = torch.prod(imp_weights, dim=-1, keepdim=True).repeat(1, num_agents)

        prod_ratios_for_surr2 = []
        for agent_idx in range(num_agents):
            ratios = imp_weights.clone()
            ratios[:,agent_idx] = torch.ones_like(ratios[:,agent_idx])
            prod_others_ratio_i = torch.prod(ratios, dim=-1).squeeze()
            inner_eps = clip_ratio_2
            prod_others_ratio_i_for_surr2 = torch.clamp(prod_others_ratio_i, 1-inner_eps, 1+inner_eps)
            prod_others_i_for_surr2 = prod_others_ratio_i_for_surr2 * imp_weights[:,agent_idx].squeeze()
            prod_ratios_for_surr2.append(prod_others_i_for_surr2)
        ratio_for_surr2 = torch.stack(prod_ratios_for_surr2, dim=-1)
        clipped_ratio = torch.clamp(ratio_for_surr2, 1.0-clip_ratio_1, 1.0+clip_ratio_1)

        # Policy loss
        surr1 = ratio * adv
        surr2 = clipped_ratio * adv
        loss_pi = -torch.min(surr1, surr2).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        pi_info = dict(kl=approx_kl)

        return loss_pi, pi_info


    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']

        return ((self.ac_model.v(obs) - ret) ** 2).mean()


    def update(self, buf, train_pi_iters, train_v_iters, target_kl, clip_ratio_1, clip_ratio_2):

        data = buf.get()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data, clip_ratio_1, clip_ratio_2)

            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                break

            loss = loss_pi
            loss.backward()
            self.loss_p_index += 1
            self.writer.add_scalar('loss/Policy_Loss', loss_pi, self.loss_p_index)
            std = torch.exp(self.ac_model.pi[0].log_std)
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
              target_kl=0.01, freeze_rep=True, entropy_coef=0.0, use_value_norm=False,
              tb_writer=None, **kargs):

        env = self.env
        self.writer = tb_writer
        self.loss_p_index, self.loss_v_index = 0, 0
        self.set_random_seed(seed)
        print('POLICY SEED', seed)

        ac_kwargs['use_beta'] = kargs['use_beta']
        self.use_beta = kargs['use_beta']

        self.setup_model(actor_critic, pi_lr, vf_lr, ac_kwargs)
        self.load_model(kargs['model_path'], kargs['cnn_model_path'], freeze_rep, steps_per_epoch)
        if self.callback:
            self.callback.init_model(self.ac_model)

        ep_ret, ep_len = 0, 0

        buf = RolloutBuffer(self.obs_dim, self.act_dim, steps_per_epoch, gamma, lam)

        if not os.path.exists(kargs['save_dir']):
            from pathlib import Path
            Path(kargs['save_dir']).mkdir(parents=True, exist_ok=True)

        step = self.start_step
        episode_lengths = []
        episode_returns = []
        episode_red_blue_damages, episode_red_red_damages, episode_blue_red_damages = [], [], []
        episode_stds = []
        last_hundred_red_blue_damages, last_hundred_red_red_damages, last_hundred_blue_red_damages = [], [], []
        best_eval_score = self.best_eval_score

        while step < steps_to_run:

            if (step + 1) % 50000 == 0 or step == 0:
                self.save_model(kargs['save_dir'], step, is_best=False)

            step += 1
            obs = torch.as_tensor(self.obs, dtype=torch.float32).to(device)

            a, v, logp, entropy = self.ac_model.step(obs)
            next_obs, r, terminal, info = env.step(a.cpu().numpy())

            ep_ret += np.sum(np.average(r, axis=0))
            ep_len += 1

            r = torch.as_tensor(r, dtype=torch.float32).to(device)
            self.obs = next_obs
            self.obs = torch.as_tensor(self.obs, dtype=torch.float32).to(device)
            buf.store(obs, a, r, v, logp)

            epoch_ended = step > 0 and step % steps_per_epoch == 0

            if np.any(terminal) or epoch_ended:
                stats = info[0]['red_stats']
                ep_rr_dmg = stats['damage_inflicted_on']['ally']
                ep_rb_dmg = stats['damage_inflicted_on']['enemy']
                ep_br_dmg = stats['damage_taken_by']['enemy']
                last_hundred_red_blue_damages.append(ep_rb_dmg)
                last_hundred_red_red_damages.append(ep_rr_dmg)
                last_hundred_blue_red_damages.append(ep_br_dmg)
                last_hundred_red_blue_damages = last_hundred_red_blue_damages[-100:]
                last_hundred_red_red_damages = last_hundred_red_red_damages[-100:]
                last_hundred_blue_red_damages = last_hundred_blue_red_damages[-100:]

                episode_lengths.append(ep_len)
                episode_returns.append(ep_ret)
                episode_red_red_damages.append(ep_rr_dmg)
                episode_blue_red_damages.append(ep_br_dmg)
                episode_red_blue_damages.append(ep_rb_dmg)
                std = torch.exp(self.ac_model.pi[0].log_std).cpu().detach().numpy()
                episode_stds.append(std)

                if epoch_ended:
                    with torch.no_grad():
                        _, v, _, _ = self.ac_model.step(self.obs)

                else:
                    with torch.no_grad():
                        v = torch.zeros((kargs['n_envs'], 5)).to(device)

                buf.finish_path(v)
                if np.any(terminal):
                    obs, ep_ret, ep_len = env.reset(), 0, 0
                    self.obs = torch.as_tensor(obs, dtype=torch.float32).to(device)

                ep_ret, ep_len, ep_rr_dmg, ep_rb_dmg, ep_br_dmg = 0, 0, 0, 0, 0

                if epoch_ended:
                    self.update(buf, train_pi_iters, train_v_iters, target_kl, clip_ratio, clip_ratio//2)


            if (step + 1) % 100 == 0:

                if self.callback:
                    self.callback.save_metrics(episode_returns, episode_lengths, episode_red_blue_damages,
                                                episode_red_red_damages, episode_blue_red_damages,
                                                episode_stds=episode_stds)

                    with open(os.path.join(self.callback.policy_record.data_dir, 'mean_statistics.json'), 'w+') as f:
                        if len(last_hundred_red_blue_damages) > 0:
                            red_red_damage = np.average(last_hundred_red_red_damages)
                            red_blue_damage = np.average(last_hundred_red_blue_damages)
                            blue_red_damage = np.average(last_hundred_blue_red_damages)
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


            if step % 50000 == 0 or step == 1:

                if self.callback and self.callback.eval_env:
                    eval_score = self.callback.validate_independent_policy(self.ac_model.state_dict(), device)
                    if eval_score > best_eval_score:
                        self.save_model(kargs['save_dir'], step, is_best=True)
                        best_eval_score = eval_score
                        with open(os.path.join(self.callback.policy_record.data_dir, 'best_eval_score.json'), 'w+') as f:
                            json.dump(best_eval_score, f)