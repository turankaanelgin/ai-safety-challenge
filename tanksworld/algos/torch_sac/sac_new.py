import pdb
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import os
from . import core


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, n_envs=1):
        self.obs_buf = torch.zeros(core.combined_shape_v3(size, n_envs, 5, obs_dim)).cuda()
        self.obs2_buf = torch.zeros(core.combined_shape_v3(size, n_envs, 5, obs_dim)).cuda()
        self.act_buf = torch.zeros(core.combined_shape_v3(size, n_envs, 5, act_dim)).cuda()
        self.rew_buf = torch.zeros((size, n_envs, 5)).cuda()
        self.done_buf = np.zeros((size, n_envs))
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class SACPolicy():

    def __init__(self, env, callback, eval_mode=False, **kargs):
        self.kargs = kargs
        self.env = env
        self.callback = callback
        self.eval_mode = eval_mode

    def run(self, num_steps):
        self.kargs.update({'epochs': num_steps//self.kargs['steps_per_epoch']})
        self.learn(**self.kargs)

    def set_random_seed(self, seed):
        # Random seed
        if seed == -1:
            MAX_INT = 2147483647
            seed = np.random.randint(MAX_INT)
        else:
            if isinstance(seed, list) and len(seed) == 1:
                seed = seed[0]

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)


    def setup_model(self, actor_critic, lr, ac_kwargs):

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.obs = self.env.reset()

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).cuda()
        self.ac_targ = deepcopy(self.ac_model)

        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.q_params = itertools.chain(self.ac_model.q1.parameters(), self.ac_model.q2.parameters())
        self.pi_optimizer = Adam(self.ac_model.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)


    def load_model(self, model_path, cnn_model_path, freeze_rep, steps_per_epoch):

        self.start_step = 0

        # Only load the representation part
        if cnn_model_path and freeze_rep:
            state_dict = torch.load(cnn_model_path)

            temp_state_dict = {}
            for key in state_dict:
                if 'cnn_net' in key:
                    temp_state_dict[key] = state_dict[key]

            self.ac_model.load_state_dict(temp_state_dict, strict=False)

            for name, param in self.ac_model.named_parameters():
                if 'cnn_net' in name:
                    param.requires_grad = False

            self.ac_targ = deepcopy(self.ac_model)


    def save_model(self, save_dir, model_id, step):

        model_path = os.path.join(save_dir, model_id, str(step) + '.pth')
        ckpt_dict = {'step': step,
                     'model_state_dict': self.ac_model.state_dict(),
                     'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),
                     'vf_optimizer_state_dict': self.q_optimizer.state_dict()}
        torch.save(ckpt_dict, model_path)


    def get_action(self, obs, deterministic=False):
        return self.ac_model.act(obs, deterministic)


    def compute_loss_q(self, data, gamma, alpha):
        obs, act, rew, obs2, done = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        obs = torch.flatten(obs, end_dim=2)
        act = torch.flatten(act, end_dim=2)
        done = torch.tile(done, (5,1)).cuda()

        q1 = self.ac_model.q1(obs, act)
        q2 = self.ac_model.q2(obs, act)

        with torch.no_grad():
            act2, logp_a2 = self.ac_model.pi(obs2)
            q1_pi_targ = self.ac_targ.q1(obs2, act2)
            q2_pi_targ = self.ac_targ.q2(obs2, act2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = rew.flatten() + gamma * (1-done).flatten() * (q_pi_targ - alpha * logp_a2.flatten())

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info


    def compute_loss_pi(self, data, alpha):
        obs = data['obs']
        pi, logp_pi = self.ac_model.pi(obs)
        q1_pi = self.ac_model.q1(obs, pi)
        q2_pi = self.ac_model.q2(obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (alpha * logp_pi.flatten() - q_pi).mean()

        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        return loss_pi, pi_info


    def update(self, data, gamma, alpha, polyak):
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data, gamma, alpha)
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.q_params:
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data, alpha)
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.q_params:
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(self.ac_model.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1-polyak) * p.data)


    def learn(self, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
              steps_per_epoch=4000, epochs=100, gamma=0.99,
              polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
              update_after=1000, update_every=50, max_ep_len=1000,
              freeze_rep=True, **kargs):

        self.set_random_seed(seed)
        self.setup_model(actor_critic, lr, ac_kwargs)
        self.load_model(kargs['model_path'], kargs['cnn_model_path'], freeze_rep, steps_per_epoch)

        if not os.path.exists(os.path.join(kargs['save_dir'], str(kargs['model_id']))):
            from pathlib import Path
            Path(os.path.join(kargs['save_dir'], str(kargs['model_id']))).mkdir(parents=True, exist_ok=True)

        ep_ret, ep_len = 0, 0
        ep_rb_dmg, ep_br_dmg, ep_rr_dmg = 0, 0, 0
        replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=steps_per_epoch)

        total_steps = steps_per_epoch * epochs
        episode_lengths = []
        episode_returns = []
        episode_red_blue_damages, episode_red_red_damages, episode_blue_red_damages = [], [], []

        self.obs = torch.as_tensor(self.obs, dtype=torch.float32).cuda()
        for t in range(self.start_step, total_steps):

            if (t + 1) % 25000 == 0 or t == 0:
                self.save_model(kargs['save_dir'], kargs['model_id'], t)

            if t > start_steps:
                action = self.get_action(self.obs)
            else:
                action = [np.expand_dims(self.env.action_space.sample(), axis=0) for _ in range(5)]
                action = np.expand_dims(np.concatenate(action, axis=0), axis=0)

            next_obs, reward, done, info = self.env.step(action)

            stats = info[0]['current']
            ep_rr_dmg += stats['red_ally_damage']
            ep_rb_dmg += stats['red_enemy_damage']
            ep_br_dmg += stats['blue_enemy_damage']
            ep_ret += np.sum(reward)
            ep_len += 1

            next_obs = torch.as_tensor(next_obs, dtype=torch.float32).cuda()
            reward = torch.as_tensor(reward, dtype=torch.float32).cuda()
            action = torch.as_tensor(action, dtype=torch.float32).cuda()

            done = False if ep_len==max_ep_len else done

            replay_buffer.store(self.obs, action, reward, next_obs, done)

            self.obs = next_obs
            self.obs = torch.as_tensor(self.obs, dtype=torch.float32).cuda()

            if done or ep_len == max_ep_len:
                episode_lengths.append(ep_len)
                episode_returns.append(ep_ret)
                episode_red_red_damages.append(ep_rr_dmg)
                episode_blue_red_damages.append(ep_br_dmg)
                episode_red_blue_damages.append(ep_rb_dmg)
                self.obs = self.env.reset()
                self.obs = torch.as_tensor(self.obs, dtype=torch.float32).cuda()
                ep_ret, ep_len, ep_rr_dmg, ep_rb_dmg, ep_br_dmg = 0, 0, 0, 0, 0

            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    self.update(data=batch, gamma=gamma, alpha=alpha, polyak=polyak)

            if t % 50 == 0 or t == 4:
                if self.callback:
                    self.callback.save_metrics(info, episode_returns, episode_lengths, episode_red_blue_damages,
                                               episode_red_red_damages, episode_blue_red_damages)
                episode_lengths = []
                episode_returns = []
                episode_red_blue_damages = []
                episode_blue_red_damages = []
                episode_red_red_damages = []
