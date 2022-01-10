import pdb
from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import os
from . import core


class ReplayBuffer:

    def __init__(self, N, obs_dim, act_dim, size):
        self.obs_buf = []
        self.obs2_buf = []
        self.act_buf = []
        self.N = N
        for i in range(N):
            self.obs_buf.append(np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32))
            self.obs2_buf.append(np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32))
            self.act_buf.append(np.zeros(core.combined_shape(size, act_dim), dtype=np.float32))
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        for i in range(self.N):
            self.obs_buf[i][self.ptr] = obs[i]
            self.obs2_buf[i][self.ptr] = next_obs[i]
            self.act_buf[i][self.ptr] = act[i]
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
                    obs=[torch.as_tensor(self.obs_buf[i][idxs], dtype=torch.float32).cuda() for i in range(self.N)],
                    obs2=[torch.as_tensor(self.obs2_buf[i][idxs], dtype=torch.float32).cuda() for i in range(self.N)],
                    act=[torch.as_tensor(self.act_buf[i][idxs], dtype=torch.float32).cuda() for i in range(self.N)],
                    rew=torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32).cuda(),
                    done=torch.as_tensor(self.done_buf[idxs], dtype=torch.float32).cuda()
                )

        return batch #{k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


class MADDPGPolicy():

    def __init__(self, env, callback, eval_mode=False, **kargs):
        self.kargs = kargs
        self.env = env
        self.callback = callback
        self.eval_mode = eval_mode

    def run(self, num_steps):
        self.kargs.update({'epochs': num_steps//self.kargs['steps_per_epoch']})
        if self.eval_mode:
            self.evaluate(steps_to_run=num_steps, model_path=self.kargs['model_path'])
        else:
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


    def setup_model(self, actor_critic, pi_lr, q_lr, ac_kwargs, common_actor):

        self.N = 5
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, common_actor, **ac_kwargs).cuda()
        self.ac_targ = deepcopy(self.ac_model)

        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.pi_optimizers = [Adam(p.parameters(), lr=pi_lr) for p in self.ac_model.unique_pis]
        self.q_optimizer = Adam(self.ac_model.q.parameters(), lr=q_lr)


    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data, gamma):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = self.ac_model.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, [self.ac_targ.pis[i](o2[i]) for i in range(self.N)])
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.cpu().detach().numpy())

        return loss_q, loss_info


    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.ac_model.q(o, [self.ac_model.pis[i](o[i]) for i in range(self.N)])
        return -q_pi.mean()


    def update(self, data, polyak, gamma):
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data, gamma)
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.ac_model.q.parameters():
            p.requires_grad = False

        [opt.zero_grad() for opt in self.pi_optimizers]
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        [opt.step() for opt in self.pi_optimizers]

        for p in self.ac_model.q.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(self.ac_model.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)


    def get_action(self, o, noise_scale):
        a = self.ac_model.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim[0])
        return np.clip(a, -1.0, 1.0)


    def save_model(self, save_dir, model_id, step):

        model_path = os.path.join(save_dir, model_id, str(step) + '.pth')
        ckpt_dict = {'step': step,
                     'model_state_dict': self.ac_model.state_dict(),
                     'pi_optimizer_state_dict': [pi_optimizer.state_dict() for pi_optimizer in self.pi_optimizers],
                     'vf_optimizer_state_dict': self.q_optimizer.state_dict()}
        torch.save(ckpt_dict, model_path)


    def learn(self, common_actor=False,
              actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
              steps_per_epoch=4000, epochs=100, replay_size=int(1e3), gamma=0.99,
              polyak=0.995, pi_lr=3e-4, q_lr=1e-3, batch_size=100, start_steps=10000,
              update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10,
              max_ep_len=1000, logger_kwargs=dict(), save_freq=1, **kargs):

        env = self.env
        self.set_random_seed(seed)
        self.setup_model(actor_critic, pi_lr, q_lr, ac_kwargs, common_actor)

        replay_buffer = ReplayBuffer(self.N, obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)

        total_steps = steps_per_epoch * epochs
        obs, ep_ret, ep_len = env.reset(), 0, 0
        ep_rb_dmg, ep_br_dmg, ep_rr_dmg = 0, 0, 0

        if not os.path.exists(os.path.join(kargs['save_dir'], str(kargs['model_id']))):
            from pathlib import Path
            Path(os.path.join(kargs['save_dir'], str(kargs['model_id']))).mkdir(parents=True, exist_ok=True)

        episode_lengths = []
        episode_returns = []
        episode_red_blue_damages, episode_red_red_damages, episode_blue_red_damages = [], [], []

        for t in range(total_steps):

            if (t + 1) % 50000 == 0 or t == 0:
                self.save_model(kargs['save_dir'], kargs['model_id'], t)

            if t > start_steps:
                obs = torch.as_tensor(obs, dtype=torch.float32).cuda()
                a = self.get_action(obs, act_noise)
            else:
                a = np.concatenate([np.expand_dims(env.action_space.sample(), axis=0) for _ in range(self.N)])
                a = np.expand_dims(a, axis=0)
            o2, rs, d, info = env.step(a)
            if self.callback:
                self.callback._on_step()

            stats = info[0]['current']
            ep_rr_dmg += stats['red_ally_damage']
            ep_rb_dmg += stats['red_enemy_damage']
            ep_br_dmg += stats['blue_enemy_damage']
            r = np.sum(rs)
            d = d[0]
            ep_ret += r
            ep_len += 1

            d = False if ep_len==max_ep_len else d
            if torch.is_tensor(obs):
                obs = obs.cpu().detach().numpy()
            replay_buffer.store(obs.squeeze(0), a.squeeze(0), r, o2.squeeze(0), d)
            obs = o2

            if d or ep_len == max_ep_len:
                episode_lengths.append(ep_len)
                episode_returns.append(ep_ret)
                episode_red_red_damages.append(ep_rr_dmg)
                episode_blue_red_damages.append(ep_br_dmg)
                episode_red_blue_damages.append(ep_rb_dmg)

                obs, ep_ret, ep_len = env.reset(), 0, 0
                ep_rr_dmg, ep_rb_dmg, ep_br_dmg = 0, 0, 0

            if t >= update_after and t % update_every == 0:
                for _ in range(update_every):
                    batch = replay_buffer.sample_batch(batch_size)
                    self.update(data=batch, polyak=polyak, gamma=gamma)

            if t % 50 == 0:
                if self.callback:
                    self.callback.save_metrics(info, episode_returns, episode_lengths, episode_red_blue_damages,
                                               episode_red_red_damages, episode_blue_red_damages)
                episode_lengths = []
                episode_returns = []
                episode_red_blue_damages = []
                episode_blue_red_damages = []
                episode_red_red_damages = []
