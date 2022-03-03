import pdb
import pprint
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from .mappo_utils.valuenorm import ValueNorm

from . import core_ind as core
import os
import json


device = torch.device('cuda')


class RolloutBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, n_envs=1):
        self.obs_buf = torch.zeros(core.combined_shape_v2(size, 5, obs_dim)).to(device)
        self.act_buf = torch.zeros(core.combined_shape_v2(size, 5, act_dim)).to(device)
        self.adv_buf = torch.zeros((size, 5)).to(device)
        self.rew_buf = torch.zeros((size, 5)).to(device)
        self.ret_buf = torch.zeros((size, 5)).to(device)
        self.val_buf = torch.zeros((size, 5)).to(device)
        self.logp_buf = torch.zeros((size, 5)).to(device)
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
        if self.eval_mode:
            ac_kwargs = {}
            ac_kwargs['use_beta'] = self.kargs['use_beta']
            ac_kwargs['local_std'] = self.kargs['local_std']
            ac_kwargs['central_critic'] = self.kargs['central_critic']
            self.evaluate(steps_to_run=num_steps, model_path=self.kargs['model_path'], ac_kwargs=ac_kwargs)
        else:
            self.learn(**self.kargs)

    def evaluate(self, steps_to_run, model_path, actor_critic=core.MLPActorCritic, ac_kwargs=dict()):

        steps = 0
        observation = self.env.reset()

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
        ckpt = torch.load(model_path)
        self.ac_model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self.ac_model.eval()
        num_envs = 10

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

        while steps < steps_to_run:
            with torch.no_grad():
                observation = torch.as_tensor(observation, dtype=torch.float32).squeeze(2).to(device)
                action, v, logp, _ = self.ac_model.step(observation)
            action = action.squeeze(0).reshape(num_envs, 5, -1)
            observation, reward, done, info = self.env.step(action.cpu().numpy())
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
                steps += 1
                observation = self.env.reset()

                if steps % 2 == 0 and steps > 0:
                    avg_red_red_damages = np.mean(episode_red_red_damages)
                    avg_red_blue_damages = np.mean(episode_red_blue_damages)
                    avg_blue_red_damages = np.mean(episode_blue_red_damages)

                    with open(os.path.join(self.callback.policy_record.data_dir, 'mean_statistics.json'), 'w+') as f:
                        json.dump({'Number of games': steps,
                                   'Red-Red-Damage': avg_red_red_damages.tolist(),
                                   'Red-Blue Damage': avg_red_blue_damages.tolist(),
                                   'Blue-Red Damage': avg_blue_red_damages.tolist()}, f, indent=4)

                    with open(os.path.join(self.callback.policy_record.data_dir, 'all_statistics.json'), 'w+') as f:
                        json.dump({'Number of games': steps,
                                   'Red-Red-Damage': all_episode_red_red_damages,
                                   'Red-Blue Damage': all_episode_red_blue_damages,
                                   'Blue-Red Damage': all_episode_blue_red_damages}, f, indent=4)

                    avg_red_red_damages_per_env = np.mean(episode_red_red_damages, axis=0)
                    avg_red_blue_damages_per_env = np.mean(episode_red_blue_damages, axis=0)
                    avg_blue_red_damages_per_env = np.mean(episode_blue_red_damages, axis=0)

                    with open(os.path.join(self.callback.policy_record.data_dir, 'mean_statistics_per_env.json'),
                              'w+') as f:
                        json.dump({'Number of games': steps,
                                   'All-Red-Red-Damage': avg_red_red_damages_per_env.tolist(),
                                   'All-Red-Blue Damage': avg_red_blue_damages_per_env.tolist(),
                                   'All-Blue-Red Damage': avg_blue_red_damages_per_env.tolist()}, f, indent=4)


    def set_random_seed(self, seed):
        # Random seed
        if seed == -1:
            MAX_INT = 2147483647
            seed = np.random.randint(MAX_INT)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def setup_model(self, actor_critic, pi_lr, vf_lr, pi_scheduler, vf_scheduler, ac_kwargs):

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.obs = self.env.reset()

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
        self.pi_optimizers = [Adam(self.ac_model.pi[idx].parameters(), lr=pi_lr) for idx in range(self.ac_model.num_agents)]
        self.vf_optimizer = Adam(self.ac_model.v.parameters(), lr=vf_lr)

    def load_model(self, model_path, cnn_model_path, freeze_rep, steps_per_epoch):

        self.start_step = 0

        if model_path:
            ckpt = torch.load(model_path)
            self.ac_model.load_state_dict(ckpt['model_state_dict'], strict=True)
            pi_ckpt = ckpt['pi_optimizer_state_dict']
            for idx, pi_opt in enumerate(self.pi_optimizers):
                pi_opt.load_state_dict(pi_ckpt[idx])
            self.vf_optimizer.load_state_dict(ckpt['vf_optimizer_state_dict'])
            self.start_step = ckpt['step']
            self.start_step -= self.start_step % steps_per_epoch

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
                     'pi_optimizer_state_dict': [self.pi_optimizers[idx].state_dict() for idx in range(self.ac_model.num_agents)],
                     'vf_optimizer_state_dict': self.vf_optimizer.state_dict()}
        torch.save(ckpt_dict, model_path)


    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data, clip_ratio, agent_idx):

        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        obs, act, adv, logp_old = obs[:,agent_idx], act[:,agent_idx], adv[:,agent_idx], logp_old[:,agent_idx]
        size = data['obs'].shape[0]

        pi, logp = self.ac_model.pi[agent_idx](obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = 0.0
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data, value_clip=False, clip_ratio=0.0):
        obs, ret = data['obs'], data['ret']
        obs = torch.flatten(obs, end_dim=1)
        ret = torch.flatten(ret)
        return ((self.ac_model.v(obs).squeeze(0) - ret) ** 2).mean()

    def update(self, buf, train_pi_iters, train_v_iters, target_kl, clip_ratio, entropy_coef, value_clip):

        data = buf.get()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            for agent_idx in range(len(self.pi_optimizers)):
                self.pi_optimizers[agent_idx].zero_grad()
                loss_pi, pi_info = self.compute_loss_pi(data, clip_ratio, agent_idx)
                kl = pi_info['kl']

                if kl > 1.5 * target_kl:
                    break

                loss_pi.backward()
                self.pi_optimizers[agent_idx].step()
            self.loss_p_index += 1

        for i in range(train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data, value_clip)
            loss_v.backward()
            self.loss_v_index += 1
            self.vf_optimizer.step()




    def learn(self, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=-1,
              steps_per_epoch=800, steps_to_run=100000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
              vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97,
              target_kl=0.01, use_value_norm=False, use_huber_loss=False, use_rnn=False, use_popart=False,
              use_sde=False, sde_sample_freq=20, pi_scheduler='cons', vf_scheduler='cons', freeze_rep=True,
              entropy_coef=0.0, kl_beta=3.0, tb_writer=None, **kargs):

        env = self.env
        self.writer = tb_writer
        self.loss_p_index, self.loss_v_index = 0, 0
        self.action_dim = 3
        self.set_random_seed(seed)
        print('POLICY SEED', seed)

        ac_kwargs['use_sde'] = use_sde
        ac_kwargs['use_rnn'] = use_rnn
        ac_kwargs['use_beta'] = kargs['use_beta']
        ac_kwargs['use_laplace'] = kargs['use_laplace']
        ac_kwargs['local_std'] = kargs['local_std']
        ac_kwargs['central_critic'] = kargs['central_critic']
        ac_kwargs['noisy'] = kargs['noisy']
        self.use_beta = kargs['use_beta']
        self.use_laplace = kargs['use_laplace']
        self.use_fixed_kl = kargs['use_fixed_kl']
        self.use_adaptive_kl = kargs['use_adaptive_kl']
        self.weight_sharing = kargs['weight_sharing']
        self.central_critic = kargs['central_critic']
        self.kl_beta = kl_beta
        self.rollback = kargs['rollback']
        self.trust_region = kargs['trust_region']

        if kargs['reward_norm']:
            reward_norm = ValueNorm(input_shape=(1,), per_element_update=True)

        self.setup_model(actor_critic, pi_lr, vf_lr, pi_scheduler, vf_scheduler, ac_kwargs)
        self.load_model(kargs['model_path'], kargs['cnn_model_path'], freeze_rep, steps_per_epoch)
        if self.callback:
            self.callback.init_model(self.ac_model)

        num_states = kargs['num_states'] if 'num_states' in kargs else None
        if use_rnn:
            assert num_states is not None
            state_history = [self.obs] * num_states
            obs = [torch.as_tensor(o, dtype=torch.float32).unsqueeze(2).to(device) for o in state_history]
            state_history = torch.cat(obs, dim=2)

        ep_ret, ep_len = 0, 0

        buf = RolloutBuffer(self.obs_dim, self.act_dim, steps_per_epoch, gamma, lam, n_envs=kargs['n_envs'],)
        self.use_sde = use_sde
        self.use_rnn = use_rnn

        if not os.path.exists(kargs['save_dir']):
            from pathlib import Path
            Path(kargs['save_dir']).mkdir(parents=True, exist_ok=True)

        step = self.start_step
        episode_lengths = []
        episode_returns = []
        episode_red_blue_damages, episode_red_red_damages, episode_blue_red_damages = [], [], []
        last_hundred_red_blue_damages, last_hundred_red_red_damages, last_hundred_blue_red_damages = [], [], []
        best_eval_score = 0

        while step < steps_to_run:

            if (step + 1) % 50000 == 0 or step == 0:
                self.save_model(kargs['save_dir'], step, is_best=False)

            if use_sde and sde_sample_freq > 0 and step % sde_sample_freq == 0:
                # Sample a new noise matrix
                self.ac_model.pi.reset_noise()

            if kargs['noisy']:
                self.ac_model.resample()

            step += 1
            if use_rnn:
                obs = torch.as_tensor(state_history, dtype=torch.float32).to(device)
            else:
                obs = torch.as_tensor(self.obs, dtype=torch.float32).to(device)

            a, v, logp, entropy = self.ac_model.step(obs)
            if use_rnn:
                a, v, logp = a.squeeze(0), v.squeeze(0), logp.squeeze(0)
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

                if epoch_ended:
                    if use_rnn:
                        with torch.no_grad():
                            _, v, _, _ = self.ac_model.step(state_history)
                            v = v.squeeze(0)
                    else:
                        with torch.no_grad():
                            _, v, _, _ = self.ac_model.step(self.obs)

                else:
                    if self.central_critic:
                        with torch.no_grad():
                            v = torch.zeros((kargs['n_envs'], 1)).to(device)
                    else:
                        with torch.no_grad():
                            v = torch.zeros((kargs['n_envs'], 5)).to(device)

                buf.finish_path(v)
                if np.any(terminal):
                    obs, ep_ret, ep_len = env.reset(), 0, 0
                    self.obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
                    # if use_rnn:
                    #    state_history = [self.obs] * num_states
                    #    obs = [torch.as_tensor(o, dtype=torch.float32).unsqueeze(2).to(device) for o in state_history]
                    #    state_history = torch.cat(obs, dim=2)

                ep_ret, ep_len, ep_rr_dmg, ep_rb_dmg, ep_br_dmg = 0, 0, 0, 0, 0

                if epoch_ended:
                    self.update(buf, train_pi_iters, train_v_iters, target_kl, clip_ratio, entropy_coef,
                                kargs['value_clip'])

            '''
            if step % 100 == 0 or step == 4:

                if self.callback:
                    self.callback.save_metrics_multienv(episode_returns, episode_lengths, episode_red_blue_damages,
                                                        episode_red_red_damages, episode_blue_red_damages)

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
            '''
            '''
            if step % 50000 == 0:

                if self.callback and self.callback.eval_env:
                    eval_score = self.callback.validate_policy(self.ac_model.state_dict(), device)
                    if eval_score > best_eval_score:
                        self.save_model(kargs['save_dir'], step, is_best=True)
                        best_eval_score = eval_score
                        with open(os.path.join(self.callback.policy_record.data_dir, 'best_eval_score.json'), 'w+') as f:
                            json.dump(best_eval_score, f)
            '''
