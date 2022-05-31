import pdb
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F

import os
import json
import pickle
import cv2
import matplotlib
import matplotlib.pyplot as plt

from tanksworld.minimap_util import *
from .heuristics import *
from . import core
from tanksworld.env import TanksWorldEnv
import gym


device = torch.device('cuda')


class RolloutBuffer:

    def __init__(self, obs_dim, act_dim, rollout_length, gamma=0.99, lam=0.95,
                 num_workers=1, centralized=False, n_agents=5):

        self.n_agents = n_agents
        self.obs_buf = np.zeros(core.combined_shape_v3(rollout_length, num_workers, self.n_agents, obs_dim))
        self.act_buf = np.zeros(core.combined_shape_v3(rollout_length, num_workers, self.n_agents, act_dim))
        self.adv_buf = np.zeros((rollout_length, num_workers, self.n_agents))
        self.rew_buf = np.zeros((rollout_length, num_workers, self.n_agents))
        self.terminal_buf = np.zeros((rollout_length, num_workers, self.n_agents))
        self.ret_buf = np.zeros((rollout_length, num_workers, self.n_agents))
        self.val_buf = np.zeros((rollout_length + 1, num_workers, self.n_agents))
        self.logp_buf = np.zeros((rollout_length, num_workers, self.n_agents))
        self.episode_starts = np.zeros((rollout_length, num_workers, self.n_agents))
        self.gamma, self.lam = gamma, lam
        self.num_workers = num_workers
        self.centralized = centralized
        self.ptr, self.rollout_length = 0, rollout_length
        self.num_workers = num_workers
        self.tensor_func = lambda x: torch.as_tensor(x, dtype=torch.float32).to(device)

    def store(self, obs, act, rew, val, logp, dones):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.rollout_length
        try:
            self.obs_buf[self.ptr] = obs.squeeze(2)
        except:
            self.obs_buf[self.ptr] = obs.squeeze(2).squeeze(2)
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.terminal_buf[self.ptr] = np.tile(np.expand_dims(dones, axis=1), (1, self.n_agents))
        self.ptr += 1

    def finish_path(self, last_val):
        ret = last_val
        self.val_buf[-1] = last_val
        adv = np.zeros((self.num_workers, 1))
        for i in reversed(range(self.rollout_length)):
            mask = (1 - self.terminal_buf[i])
            ret = self.rew_buf[i] + self.gamma * mask * ret
            td_error = self.rew_buf[i] + self.gamma * mask * self.val_buf[i + 1] - self.val_buf[i]
            adv = adv * self.lam * self.gamma * mask + td_error
            self.adv_buf[i] = adv
            self.ret_buf[i] = ret

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        self.ptr = 0

        adv = self.tensor_func(self.adv_buf).flatten(end_dim=1)
        adv_std, adv_mean = torch.std_mean(adv)
        adv = (adv - adv_mean) / adv_std

        obs, ret, logp, val, act = [self.tensor_func(x).flatten(end_dim=1) for x in
                                    [self.obs_buf, self.ret_buf, self.logp_buf, self.val_buf, self.act_buf]
                                    ]

        return dict(obs=obs.detach(), adv=adv.detach(), logp=logp.detach(), val=val.detach(), ret=ret.detach(), act=act.detach())


class PPOPolicy():

    def __init__(self, env, callback, eval_mode=False, visual_mode=False, data_mode=False, **kargs):
        self.kargs = kargs
        self.env = env
        self.callback = callback
        self.eval_mode = eval_mode
        self.visual_mode = visual_mode
        self.data_mode = data_mode

    def run(self, num_steps):
        self.kargs.update({'steps_to_run': num_steps})

        ac_kwargs = {}
        ac_kwargs['init_log_std'] = self.kargs['init_log_std']
        ac_kwargs['centralized'] = self.kargs['centralized']
        ac_kwargs['centralized_critic'] = self.kargs['centralized_critic']
        ac_kwargs['local_std'] = self.kargs['local_std']
        #ac_kwargs['num_agents'] = 5 if self.kargs['env_name'] == 'tanksworld' else 1

        if self.eval_mode:
            self.evaluate(episodes_to_run=num_steps, model_path=self.kargs['model_path'],
                          num_envs=self.kargs['n_envs'], ac_kwargs=ac_kwargs)
        elif self.visual_mode:
            self.visualize(episodes_to_run=num_steps, model_path=self.kargs['model_path'],
                           env_name=self.kargs['env_name'], ac_kwargs=ac_kwargs)
        elif self.data_mode:
            self.collect_data(episodes_to_run=num_steps, model_path=self.kargs['model_path'], ac_kwargs=ac_kwargs)
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


    def setup_model(self, actor_critic, pi_lr, vf_lr, ac_kwargs, enemy_model=None):

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.obs = self.env.reset()
        self.state_vector = np.zeros((12, 6))

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
        if self.shared_rep:
            parameters = list(self.ac_model.pi.parameters()) + list(self.ac_model.v.parameters())
            self.optimizer = Adam(parameters, lr=pi_lr)
        else:
            self.pi_optimizer = Adam(self.ac_model.pi.parameters(), lr=pi_lr)
            self.vf_optimizer = Adam(self.ac_model.v.parameters(), lr=vf_lr)

        if self.rnd:
            self.rnd_network = core.RNDNetwork().cuda()
            self.rnd_pred_network = core.RNDNetwork().cuda()
            self.rnd_optimizer = Adam(self.rnd_pred_network.parameters(), 1e-5)
            self.rnd_network.requires_grad = False

        if self.selfplay:
            self.enemy_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
            self.enemy_model.requires_grad = False
            self.enemy_model.eval()
        elif enemy_model is not None:
            self.enemy_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
            self.enemy_model.load_state_dict(torch.load(enemy_model)['model_state_dict'], strict=True)
            self.enemy_model.requires_grad = False
            self.enemy_model.eval()


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

            if self.rnd:
                rnd_state_dict = {}
                for key in temp_state_dict:
                    if key.startswith('pi'):
                        rnd_state_dict[key[3:]] = temp_state_dict[key]

                self.rnd_network.load_state_dict(rnd_state_dict, strict=False)
                self.rnd_pred_network.load_state_dict(rnd_state_dict, strict=False)

        if freeze_rep:
            for name, param in self.ac_model.named_parameters():
                if 'cnn_net' in name:
                    param.requires_grad = False
            if self.rnd:
                for name, param in self.rnd_pred_network.named_parameters():
                    if 'cnn_net' in name:
                        param.requires_grad = False

        from torchinfo import summary
        summary(self.ac_model)
        if self.rnd:
            summary(self.rnd_pred_network)


    def save_model(self, save_dir, step, is_best=False):

        if is_best:
            model_path = os.path.join(save_dir, 'best.pth')
        else:
            model_path = os.path.join(save_dir, str(step) + '.pth')
        if self.shared_rep:
            ckpt_dict = {'step': step,
                         'model_state_dict': self.ac_model.state_dict(),
                         'optimizer_state_dict': self.optimizer.state_dict()}
        else:
            ckpt_dict = {'step': step,
                        'model_state_dict': self.ac_model.state_dict(),
                        'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),
                        'vf_optimizer_state_dict': self.vf_optimizer.state_dict()}
        if self.selfplay:
            ckpt_dict['enemy_model_state_dict'] = self.enemy_model.state_dict()
        torch.save(ckpt_dict, model_path)


    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data, clip_ratio):

        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.ac_model.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item() if pi.entropy() is not None else 0.0
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']

        values = self.ac_model.v(obs)
        return ((values - ret) ** 2).mean()

    def compute_loss_entropy(self, data):
        logp = data['logp']
        return -torch.mean(-logp)

    def random_sample1(self, indices, batch_size):
        indices = np.asarray(np.random.permutation(indices))
        batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
        for batch in batches:
            yield batch
        r = len(indices) % batch_size
        if r:
            yield indices[-r:]

    def update(self, buf, train_pi_iters, train_v_iters, target_kl, clip_ratio, entropy_coef):

        data = buf.get()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            for batch_indices in self.random_sample1(np.arange(data['obs'].shape[0]), self.batch_size):
                batch_data = {key: value[batch_indices] for key, value in data.items()}
                loss_pi, pi_info = self.compute_loss_pi(batch_data, clip_ratio)

                if self.shared_rep:
                    self.optimizer.zero_grad()

                kl = pi_info['kl']
                if kl <= 1.5 * target_kl:
                    if not self.shared_rep:
                        self.pi_optimizer.zero_grad()
                    loss_pi.backward()
                    if not self.shared_rep:
                        self.pi_optimizer.step()

                self.loss_p_index += 1
                self.writer.add_scalar('loss/Policy_Loss', loss_pi, self.loss_p_index)
                if entropy_coef > 0.0:
                    self.writer.add_scalar('loss/Entropy_Loss', loss_entropy, self.loss_p_index)
                std = torch.exp(self.ac_model.pi.log_std) if not self.discrete_action else torch.zeros((3))
                self.writer.add_scalar('std_dev/move', std[0].item(), self.loss_p_index)
                self.writer.add_scalar('std_dev/turn', std[1].item(), self.loss_p_index)
                self.writer.add_scalar('std_dev/shoot', std[2].item(), self.loss_p_index)

                loss_v = self.compute_loss_v(batch_data)
                if not self.shared_rep:
                    self.vf_optimizer.zero_grad()
                loss_v.backward()
                if not self.shared_rep:
                    self.vf_optimizer.step()
                else:
                    self.optimizer.step()
                self.loss_v_index += 1
                self.writer.add_scalar('loss/Value_Loss', loss_v, self.loss_v_index)


    def learn(self, actor_critic=core.ActorCritic, ac_kwargs=dict(), seed=-1,
              rollout_length=2048, batch_size=64, steps_to_run=100000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
              vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97,
              target_kl=0.01, freeze_rep=True, entropy_coef=0.0, use_value_norm=False,
              tb_writer=None, selfplay=False, ally_heuristic=False, enemy_heuristic=False, dense_reward=False,
              centralized=False, centralized_critic=False, local_std=False, enemy_model=None, single_agent=False,
              discrete_action=False, rnd=False, noisy=False, shared_rep=False, rnd_bonus=0.0, **kargs):

        env = self.env
        self.writer = tb_writer
        self.loss_p_index, self.loss_v_index = 0, 0
        self.set_random_seed(seed)

        shared_rep = True

        ac_kwargs['init_log_std'] = kargs['init_log_std']
        ac_kwargs['centralized'] = centralized
        ac_kwargs['centralized_critic'] = centralized_critic
        ac_kwargs['local_std'] = local_std
        ac_kwargs['discrete_action'] = discrete_action
        ac_kwargs['noisy'] = noisy
        ac_kwargs['shared_rep'] = shared_rep
        #ac_kwargs['num_agents'] = 5 if kargs['env_name'] == 'tanksworld' else 1

        self.centralized = centralized
        self.centralized_critic = centralized_critic
        self.selfplay = selfplay
        self.single_agent = single_agent
        self.discrete_action = discrete_action
        self.rnd = rnd
        self.rnd_bonus = rnd_bonus
        self.shared_rep = shared_rep
        self.batch_size = batch_size

        self.tensor_func = lambda x: torch.as_tensor(x, dtype=torch.float32).to(device)
        print('POLICY SEED', seed)

        self.prev_ckpt = None
        self.setup_model(actor_critic, pi_lr, vf_lr, ac_kwargs, enemy_model=enemy_model)
        self.load_model(kargs['model_path'], kargs['cnn_model_path'], freeze_rep, rollout_length)
        num_envs = kargs['n_envs']
        if self.callback:
            self.callback.init_model(self.ac_model)

        ep_ret = 0
        ep_intrinsic_ret = 0
        ep_len = 0
        ep_rb_dmg = np.zeros(num_envs)
        ep_br_dmg = np.zeros(num_envs)
        ep_rr_dmg = np.zeros(num_envs)

        buf = RolloutBuffer(self.obs_dim, self.act_dim, rollout_length, gamma,
                            lam, num_workers=num_envs, centralized=centralized,
                            n_agents=5)

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

        obs = self.obs

        while step < steps_to_run:

            if (step + 1) % 50000 == 0 or step == 0:  # Periodically save the model
                self.save_model(kargs['save_dir'], step)

            step += 1

            if enemy_model is not None:
                ally_obs = obs[:, :5, :, :, :]
                ally_a, v, logp, entropy = self.ac_model.step(self.tensor_func(ally_obs))
                enemy_obs = obs[:, 5:, :, :, :]
                with torch.no_grad():
                    enemy_a, _, _, _ = self.enemy_model.step(self.tensor_func(enemy_obs))
                a = torch.cat((ally_a, enemy_a), dim=1)

            else:
                a, v, logp, entropy = self.ac_model.step(self.tensor_func(obs))

            a, v, logp = a.cpu().numpy(), v.cpu().numpy(), logp.cpu().numpy()
            next_obs, r, terminal, info = env.step(a)

            ep_ret += np.average(np.sum(r, axis=1))
            ep_len += 1

            if enemy_model is not None:
                obs = np.expand_dims(ally_obs, axis=2)
                r = r[:, :5]
                a = ally_a.cpu().numpy()

            buf.store(obs, a, r, v, logp, terminal)
            obs = next_obs

            for env_idx, done in enumerate(terminal):
                if done:
                    stats = info[env_idx]['red_stats']
                    ep_rr_dmg[env_idx] = stats['damage_inflicted_on']['ally']
                    ep_rb_dmg[env_idx] = stats['damage_inflicted_on']['enemy']
                    ep_br_dmg[env_idx] = stats['damage_taken_by']['enemy']
                    episode_red_red_damages.append(ep_rr_dmg[env_idx])
                    episode_blue_red_damages.append(ep_br_dmg[env_idx])
                    episode_red_blue_damages.append(ep_rb_dmg[env_idx])
                    last_hundred_red_blue_damages[env_idx].append(ep_rb_dmg[env_idx])
                    last_hundred_red_red_damages[env_idx].append(ep_rr_dmg[env_idx])
                    last_hundred_blue_red_damages[env_idx].append(ep_br_dmg[env_idx])
                    last_hundred_red_blue_damages[env_idx] = last_hundred_red_blue_damages[env_idx][-100:]
                    last_hundred_red_red_damages[env_idx] = last_hundred_red_red_damages[env_idx][-100:]
                    last_hundred_blue_red_damages[env_idx] = last_hundred_blue_red_damages[env_idx][-100:]

            epoch_ended = step > 0 and step % batch_size == 0

            if epoch_ended:
                buf.finish_path(v)
                self.update(buf, train_pi_iters, train_v_iters, target_kl, clip_ratio, entropy_coef)

                episode_lengths.append(ep_len)
                episode_returns.append(ep_ret)

                ep_ret = 0
                ep_len = 0

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