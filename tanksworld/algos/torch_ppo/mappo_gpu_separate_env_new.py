import pdb
import pprint
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CyclicLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from . import core
import os
import json
from matplotlib import pyplot as plt


device = torch.device('cuda')


class RolloutBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, n_envs=1, use_sde=False,
                 use_rnn=False, n_states=3):

        if use_rnn:
            self.obs_buf = torch.zeros(core.combined_shape_v4(size, n_envs, 5, n_states, obs_dim)).to(device)
        else:
            self.obs_buf = torch.zeros(core.combined_shape_v3(size, n_envs, 5, obs_dim)).to(device)
        self.act_buf = torch.zeros(core.combined_shape_v3(size, n_envs, 5, act_dim)).to(device)
        self.adv_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.rew_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.ret_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.val_buf = torch.zeros((size, n_envs, 5)).to(device)
        self.logp_buf = torch.zeros((size, n_envs, 5)).to(device)
        if not use_sde:
            self.entropy_buf = torch.zeros(core.combined_shape_v3(size, n_envs, 5, act_dim)).to(device)
        else:
            self.entropy_buf = torch.zeros(core.combined_shape_v2(size, n_envs, 5)).to(device)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.n_envs = n_envs
        self.use_rnn = use_rnn

    def store(self, obs, act, rew, val, logp, entropy):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """

        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs.squeeze(2) if not self.use_rnn else obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.entropy_buf[self.ptr] = entropy if entropy is not None else torch.Tensor([0])
        self.ptr += 1

    def finish_path(self, last_val=[0, 0, 0, 0, 0]):
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

        last_val = last_val.unsqueeze(0)
        if self.use_rnn and len(last_val.shape) == 2:
            last_val = last_val.unsqueeze(0)
        rews = torch.cat((self.rew_buf[path_slice], last_val), dim=0)
        vals = torch.cat((self.val_buf[path_slice], last_val), dim=0)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        discount_delta = core.discount_cumsum(deltas.cpu().numpy(), self.gamma * self.lam)
        self.adv_buf[path_slice] = torch.as_tensor(discount_delta.copy(), dtype=torch.float32).to(device)

        # the next line computes rewards-to-go, to be targets for the value function
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
        adv_buf = self.adv_buf.flatten(start_dim=1)
        adv_std, adv_mean = torch.std_mean(adv_buf, dim=0)
        adv_buf = (adv_buf - adv_mean) / adv_std
        self.adv_buf = adv_buf.reshape(adv_buf.shape[0], self.n_envs, 5)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, entropy=self.entropy_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in data.items()}



class PPOPolicy():

    def __init__(self, env, callback, eval_mode=False, **kargs):
        self.kargs = kargs
        self.env = env
        self.callback = callback
        self.eval_mode = eval_mode


    def run(self, num_steps):
        self.kargs.update({'steps_to_run': num_steps})
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


    def setup_model(self, actor_critic, pi_lr, vf_lr, ac_kwargs):
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.obs = self.env.reset()

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
        self.pi_optimizer = Adam(self.ac_model.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac_model.v.parameters(), lr=vf_lr)


    def load_model(self, model_path, cnn_model_path, freeze_rep, steps_per_epoch):

        self.start_step = 0
        # Load from previous checkpoint
        if model_path:
            ckpt = torch.load(model_path)
            self.ac_model.load_state_dict(ckpt['model_state_dict'], strict=True)
            self.pi_optimizer.load_state_dict(ckpt['pi_optimizer_state_dict'])
            self.vf_optimizer.load_state_dict(ckpt['vf_optimizer_state_dict'])
            self.start_step = ckpt['step']
            self.start_step -= self.start_step % steps_per_epoch

            if freeze_rep:
                for name, param in self.ac_model.named_parameters():
                    if 'cnn_net' in name:
                        param.requires_grad = False

        # Only load the representation part
        elif cnn_model_path:
            state_dict = torch.load(cnn_model_path)

            temp_state_dict = {}
            for key in state_dict:
                if 'cnn_net' in key:
                    temp_state_dict[key] = state_dict[key]

            self.ac_model.load_state_dict(temp_state_dict, strict=False)

        elif freeze_rep:
            for name, param in self.ac_model.named_parameters():
                if 'cnn_net' in name:
                    param.requires_grad = False


    def save_model(self, save_dir, model_id, step):

        model_path = os.path.join(save_dir, model_id, str(step) + '.pth')
        ckpt_dict = {'step': step,
                     'model_state_dict': self.ac_model.state_dict(),
                     'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),
                     'vf_optimizer_state_dict': self.vf_optimizer.state_dict()}
        torch.save(ckpt_dict, model_path)


    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data, clip_ratio):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        size = data['obs'].shape[0]
        obs, act, adv, logp_old = obs.to(device), act.to(device), adv.to(device), logp_old.to(device)
        obs = torch.flatten(obs, end_dim=2)
        act = torch.flatten(act, end_dim=2)

        # Policy loss
        pi, logp = self.ac_model.pi(obs, act)
        logp = logp.reshape(size, -1, 5)
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
        obs, ret = obs.to(device), ret.to(device)
        obs = torch.flatten(obs, end_dim=2)
        ret = torch.flatten(ret)
        return ((self.ac_model.v(obs).squeeze(0) - ret) ** 2).mean()


    def compute_loss_entropy(self, data):
        logp = data['logp']
        return -torch.mean(-logp)


    def update(self, buf, train_pi_iters, train_v_iters, target_kl, clip_ratio, entropy_coef):

        data = []
        for env_idx in range(len(buf)):
            data.append(buf[env_idx].get())
        pdb.set_trace()
        combined_data = dict()
        for key in data[0]:
            combined_data[key] = torch.cat([data[i][key].unsqueeze(1) for i in range(len(data))], dim=1)
        data = combined_data

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data, clip_ratio)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                # logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            if entropy_coef > 0.0:
                loss_entropy = self.compute_loss_entropy(data)
                loss = loss_pi + entropy_coef * loss_entropy
            else:
                loss = loss_pi
            loss.backward()
            self.loss_p_index += 1
            self.writer.add_scalar('loss/Policy_Loss', loss_pi, self.loss_p_index)
            if entropy_coef > 0.0:
                self.writer.add_scalar('loss/Entropy_Loss', loss_entropy, self.loss_p_index)
            std = torch.exp(self.ac_model.pi.log_std)
            self.writer.add_scalar('std_dev/move', std[0].item(), self.loss_p_index)
            self.writer.add_scalar('std_dev/turn', std[1].item(), self.loss_p_index)
            self.writer.add_scalar('std_dev/shoot', std[2].item(), self.loss_p_index)
            if self.use_rnn:
                torch.nn.utils.clip_grad_norm(self.ac_model.parameters(), 0.5)
            self.pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.loss_v_index += 1
            self.writer.add_scalar('loss/Value_Loss', loss_v, self.loss_v_index)
            self.vf_optimizer.step()


    def learn(self, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=-1,
              steps_per_epoch=800, steps_to_run=100000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
              vf_lr=1e-3, ent_coef=0.0, train_pi_iters=80, train_v_iters=80, lam=0.97,
              target_kl=0.01, tsboard_freq=-1, curriculum_start=-1, curriculum_stop=-1, use_value_norm=False,
              use_huber_loss=False, use_rnn=False, use_popart=False, use_sde=False, sde_sample_freq=1,
              pi_scheduler='cons', vf_scheduler='cons', freeze_rep=True, entropy_coef=0.0,
              tb_writer=None, **kargs):

        env = self.env
        self.writer = tb_writer
        self.loss_p_index, self.loss_v_index = 0, 0
        self.set_random_seed(seed)
        self.setup_model(actor_critic, pi_lr, vf_lr, ac_kwargs)
        self.load_model(kargs['model_path'], kargs['cnn_model_path'], freeze_rep, steps_per_epoch)

        ep_ret, ep_len = 0, 0
        ep_rb_dmg, ep_br_dmg, ep_rr_dmg = 0, 0, 0
        ep_ret_scheduler, ep_len_scheduler = 0, 0

        buf = []
        for _ in range(kargs['n_envs']):
            buf.append(RolloutBuffer(self.obs_dim, self.act_dim, steps_per_epoch, gamma,
                                     lam, n_envs=kargs['n_envs'], use_sde=use_sde, use_rnn=use_rnn,
                                     n_states=num_states))
        self.use_sde = use_sde
        self.use_rnn = use_rnn

        if not os.path.exists(os.path.join(kargs['save_dir'], str(kargs['model_id']))):
            from pathlib import Path
            Path(os.path.join(kargs['save_dir'], str(kargs['model_id']))).mkdir(parents=True, exist_ok=True)

        step = self.start_step
        episode_lengths = []
        episode_returns = []
        episode_red_blue_damages, episode_red_red_damages, episode_blue_red_damages = [], [], []

        while step < steps_to_run:

            if (step + 1) % 50000 == 0:
                self.save_model(kargs['save_dir'], kargs['model_id'], step)

            step += 1
            obs = torch.as_tensor(self.obs, dtype=torch.float32).to(device)
            a, v, logp, entropy = self.ac_model.step(obs)
            next_obs, r, terminal, info = env.step(a.cpu().numpy())
            if self.callback:
                self.callback._on_step()

            stats = info[0]['current']
            ep_rr_dmg += stats['red_ally_damage']
            ep_rb_dmg += stats['red_enemy_damage']
            ep_br_dmg += stats['blue_enemy_damage']
            ep_ret += np.sum(r)
            ep_ret_scheduler += np.sum(r)
            ep_len += 1
            ep_len_scheduler += 1

            r = torch.as_tensor(r, dtype=torch.float32).to(device)

            self.obs = next_obs
            buf.store(obs, a, r, v, logp, entropy)
            epoch_ended = step > 0 and step % steps_per_epoch == 0

            if epoch_ended:

                episode_lengths.append(ep_len)
                episode_returns.append(ep_ret)
                episode_red_red_damages.append(ep_rr_dmg)
                episode_blue_red_damages.append(ep_br_dmg)
                episode_red_blue_damages.append(ep_rb_dmg)

                _, v, _, _ = self.ac_model.step(
                    torch.as_tensor(self.obs, dtype=torch.float32).to(device))
                for env_idx in range(len(buf)):
                    buf[env_idx].finish_path(v[env_idx])

                self.update(buf, train_pi_iters, train_v_iters, target_kl, clip_ratio, entropy_coef)
                ep_ret, ep_len, ep_rr_dmg, ep_rb_dmg, ep_br_dmg = 0, 0, 0, 0, 0

            else:

                for env_idx, term in enumerate(terminal):
                    if term:
                        v = torch.zeros((5,)).to(device)
                        buf[env_idx].finish_path(v)
                        local_o = env.reset_single_env(env_idx)
                        local_o = np.expand_dims(local_o, axis=0)
                        if env_idx == 0:
                            self.obs = np.concatenate((local_o, self.obs[env_idx+1:]), axis=0)
                        elif env_idx == len(buf)-1:
                            self.obs = np.concatenate((self.obs[:env_idx], local_o), axis=0)
                        else:
                            self.obs = np.concatenate((self.obs[:env_idx], local_o, o[env_idx+1:]), axis=0)

            if step % 50 == 0:
                if self.callback:
                    self.callback.save_metrics(info, episode_returns, episode_lengths, episode_red_blue_damages,
                                               episode_red_red_damages, episode_blue_red_damages)
                episode_lengths = []
                episode_returns = []
                episode_red_blue_damages = []
                episode_blue_red_damages = []
                episode_red_red_damages = []
