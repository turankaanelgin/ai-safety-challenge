import copy
import pdb
import numpy as np
import scipy
import torch
from torch.optim import Adam
import torch.nn.functional as F

import os
import json
import random
from sklearn.kernel_approximation import RBFSampler

from tanksworld.minimap_util import *
from . import core
from .utils.normalizer import *


device = torch.device('cuda')


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


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
        self.obs_buf[self.ptr] = obs
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
        if torch.any(torch.isnan(adv_std)): pdb.set_trace()
        adv_buf = (adv_buf - adv_mean) / adv_std
        self.adv_buf = adv_buf.reshape(adv_buf.shape[0], self.n_rollout_threads, self.n_agents)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf, rew=self.rew_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in data.items()}


class PCPGPolicy():

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
        if self.eval_mode:
            self.evaluate(episodes_to_run=num_steps, model_path=self.kargs['model_path'],
                          num_envs=self.kargs['n_envs'], ac_kwargs=ac_kwargs)
        else:
            self.learn(**self.kargs)

    def learn(self, **config):

        self.config = config

        self.epoch = 0
        self.set_random_seed(config['seed'])
        if not os.path.exists(config['save_dir']):
            from pathlib import Path
            Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
        self.setup_model(core.ActorCritic, config['lr'], dict())
        self.load_representation()

        self.ep_ret = 0
        self.ep_ret_intrinsic = 0
        self.ep_len = 0
        self.ep_rb_dmg = np.zeros(config['num_envs'])
        self.ep_br_dmg = np.zeros(config['num_envs'])
        self.ep_rr_dmg = np.zeros(config['num_envs'])

        self.episode_lengths = []
        self.episode_returns = []
        self.episode_intrinsic_returns = []
        self.episode_red_blue_damages, self.episode_red_red_damages, self.episode_blue_red_damages = [], [], []
        self.episode_stds = []
        # Damage for last hundred steps
        self.last_hundred_red_blue_damages = [[] for _ in range(config['num_envs'])]
        self.last_hundred_red_red_damages = [[] for _ in range(config['num_envs'])]
        self.last_hundred_blue_red_damages = [[] for _ in range(config['num_envs'])]
        self.best_eval_score = 0

        while True:
            total_episodes = self.total_steps / config['horizon']
            if self.total_steps % 50000 == 0:
                self.save(save_dir=config['save_dir'], step=self.total_steps)

            if self.epoch == config['start_exploit']:
                self.initialize_new_policy('exploit')
            self.update_replay_buffer()
            self.update_density_model(mode='explore-exploit')
            self.optimize_policy()
            self.epoch += 1

            '''
            REWARD
            '''

    def save(self, save_dir, step, is_best=False):

        if is_best:
            model_path = os.path.join(save_dir, 'best.pth')
        else:
            model_path = os.path.join(save_dir, str(step) + '.pth')
        ckpt_dict = {'step': self.total_steps,
                     'model_explore_exploit_state_dict': self.network['explore-exploit'].state_dict(),
                     'model_exploit_state_dict': self.network['exploit'].state_dict(),
                     'model_rollin_state_dict': self.network['rollin'].state_dict(),
                     'optimizer_explore_exploit_state_dict': self.optimizer['explore-exploit'].state_dict(),
                     'policy_mixture_weights': self.policy_mixture_weights,}
        torch.save(ckpt_dict, model_path)

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

    def setup_model(self, actor_critic, pi_lr, ac_kwargs):

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.states = self.env.reset()

        self.network = dict()
        self.optimizer = dict()
        self.replay_buffer = dict()
        self.density_model = dict()
        self.replay_buffer_actions = dict()

        for mode in ['explore-exploit', 'rollin']:
            self.network[mode] = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
            self.replay_buffer[mode] = []
            self.replay_buffer_actions[mode] = []

        self.optimizer['explore-exploit'] = Adam(self.network['explore-exploit'].parameters(), lr=pi_lr)

        self.network['exploit'] = self.network['explore-exploit']
        self.total_steps = 0

        self.policy_mixture = [copy.deepcopy(self.network['explore-exploit'].state_dict())]
        self.policy_mixture_optimizers = [copy.deepcopy(self.optimizer['explore-exploit'].state_dict())]
        self.policy_mixture_weights = torch.tensor([1.0])
        self.policy_mixture_returns = []

        self.rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.config['phi_dim'])
        self.rbf_feature.fit(X=np.random.randn(5, 9216+3))

        self.uniform_prob = self.continuous_uniform_prob()

    def load_representation(self):

        state_dict = torch.load(self.config['cnn_model_path'])

        temp_state_dict = {}
        for key in state_dict:
            if 'cnn_net' in key:
                temp_state_dict[key] = state_dict[key]

        for mode in ['explore-exploit', 'exploit', 'rollin']:
            self.network[mode].load_state_dict(temp_state_dict, strict=False)

        if self.config['freeze_rep']:
            for mode in ['explore-exploit', 'exploit', 'rollin']:
                for name, param in self.network[mode].named_parameters():
                    if 'cnn_net' in name:
                        param.requires_grad = False

    def compute_reward_bonus(self, states, actions):

        with torch.no_grad():
            states_reshaped = torch.flatten(states, end_dim=1).cuda()
            states = self.network['exploit'].pi.cnn_net(states_reshaped)
        phi = self.compute_kernel(states, actions)
        reward_bonus = torch.sqrt((torch.mm(phi, self.density_model) * phi).sum(1)).detach()
        return reward_bonus

    def gather_trajectories(self, roll_in=True, add_bonus_reward=True, mode=None, record_return=False):

        states = self.states
        network = self.network[mode]

        roll_in_length = 0 if not roll_in else random.randint(0, self.config['horizon']-5)
        roll_out_length = self.config['horizon'] - roll_in_length
        buffer = RolloutBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=roll_out_length)

        total_rewards = {'real': 0.0, 'bonus': 0.0}

        if roll_in_length > 0:
            assert roll_in
            i = torch.multinomial(self.policy_mixture_weights.cpu(), num_samples=1)
            self.network['rollin'].load_state_dict(self.policy_mixture[i])

            for _ in range(roll_in_length):
                actions, v, logp, entropy = self.network['rollin'].step(torch.as_tensor(states, dtype=torch.float32).cuda())
                next_states, rewards, terminals, info = self.env.step(actions.cpu().numpy())
                states = next_states
                self.total_steps += 1

        for i in range(roll_out_length):
            if i == 0 and roll_in:
                sample_eps_greedy = random.random() < self.config['eps']
                if sample_eps_greedy:
                    actions = self.uniform_sample_cont_random_acts(states.shape[0])
                    actions = torch.as_tensor(actions, dtype=torch.float32).cuda()
                    pi, logp = network.pi(torch.as_tensor(states, dtype=torch.float32).cuda(),
                                          actions)
                    values = network.v(torch.as_tensor(states, dtype=torch.float32).cuda())
                else:
                    actions, values, logp, _ = network.step(torch.as_tensor(states, dtype=torch.float32).to(device))
                logp = (logp.exp() * (1-self.config['eps']) + self.config['eps']*self.uniform_prob).log()
            else:
                actions, values, logp, _ = network.step(torch.as_tensor(states, dtype=torch.float32).to(device))

            next_states, rewards, terminals, info = self.env.step(actions.cpu().numpy())

            if add_bonus_reward:
                reward_bonus = self.config['reward_bonus_normalizer'](self.compute_reward_bonus(
                    torch.as_tensor(states, dtype=torch.float32), actions))
                reward_bonus = reward_bonus * self.config['horizon']
                if not roll_in:
                    total_rewards['bonus'] += reward_bonus.mean().item()
                    total_rewards['real'] += rewards.mean()
                rewards += self.config['bonus_coeff'] * reward_bonus.cpu().numpy()

            if random.random() < 0.01 and not roll_in:
                ri = total_rewards['bonus'] * self.config['bonus_coeff']
                re = total_rewards['real']
            else:
                ri = self.config['bonus_coeff'] * reward_bonus.cpu().numpy().mean() if add_bonus_reward else 0.0
                re = (rewards - self.config['bonus_coeff'] * reward_bonus.cpu().numpy()).mean() if add_bonus_reward else rewards.mean()

            if record_return:

                self.ep_ret += re
                self.ep_ret_intrinsic += ri
                self.ep_len += 1

                for env_idx, done in enumerate(terminals):
                    if done:
                        stats = info[env_idx]['red_stats']
                        self.ep_rr_dmg[env_idx] = stats['damage_inflicted_on']['ally']
                        self.ep_rb_dmg[env_idx] = stats['damage_inflicted_on']['enemy']
                        self.ep_br_dmg[env_idx] = stats['damage_taken_by']['enemy']
                        self.last_hundred_red_blue_damages[env_idx].append(self.ep_rb_dmg[env_idx])
                        self.last_hundred_red_red_damages[env_idx].append(self.ep_rr_dmg[env_idx])
                        self.last_hundred_blue_red_damages[env_idx].append(self.ep_br_dmg[env_idx])
                        self.last_hundred_red_blue_damages[env_idx] = self.last_hundred_red_blue_damages[env_idx][-100:]
                        self.last_hundred_red_red_damages[env_idx] = self.last_hundred_red_red_damages[env_idx][-100:]
                        self.last_hundred_blue_red_damages[env_idx] = self.last_hundred_blue_red_damages[env_idx][-100:]

                if np.any(terminals) or i == roll_out_length-1:
                    self.episode_lengths.append(self.ep_len)
                    self.episode_returns.append(self.ep_ret)
                    self.episode_intrinsic_returns.append(self.ep_ret_intrinsic)
                    self.episode_red_red_damages.append(self.ep_rr_dmg)
                    self.episode_blue_red_damages.append(self.ep_br_dmg)
                    self.episode_red_blue_damages.append(self.ep_rb_dmg)
                    self.episode_stds.append([0, 0, 0])

                    self.ep_ret = 0
                    self.ep_ret_intrinsic = 0
                    self.ep_len = 0
                    self.ep_rb_dmg = np.zeros(self.config['num_envs'])
                    self.ep_br_dmg = np.zeros(self.config['num_envs'])
                    self.ep_rr_dmg = np.zeros(self.config['num_envs'])

                if i == roll_out_length-1:

                    if self.callback:
                        assert len(self.episode_returns) == len(self.episode_lengths) == len(self.episode_intrinsic_returns)
                        self.callback.save_metrics_multienv(self.episode_returns, self.episode_lengths,
                                                            self.episode_red_blue_damages,
                                                            self.episode_red_red_damages, self.episode_blue_red_damages,
                                                            episode_stds=[],
                                                            episode_intrinsic_rewards=self.episode_intrinsic_returns)

                        with open(os.path.join(self.callback.policy_record.data_dir, 'mean_statistics.json'),
                                  'w+') as f:
                            if self.last_hundred_red_blue_damages[0] is not None:
                                red_red_damage = np.average(np.concatenate(self.last_hundred_red_red_damages))
                                red_blue_damage = np.average(np.concatenate(self.last_hundred_red_blue_damages))
                                blue_red_damage = np.average(np.concatenate(self.last_hundred_blue_red_damages))
                            else:
                                red_red_damage, red_blue_damage, blue_red_damage = 0.0, 0.0, 0.0

                            json.dump({'Red-Blue-Damage': red_blue_damage,
                                       'Red-Red-Damage': red_red_damage,
                                       'Blue-Red-Damage': blue_red_damage}, f, indent=True)

                    self.episode_lengths = []
                    self.episode_returns = []
                    self.episode_intrinsic_returns = []
                    self.episode_red_blue_damages = []
                    self.episode_blue_red_damages = []
                    self.episode_red_red_damages = []
                    self.episode_stds = []
                    total_rewards = {'real': 0.0, 'bonus': 0.0}

            rewards = self.config['reward_normalizer'](rewards)
            buffer.store(torch.as_tensor(states, dtype=torch.float32),
                         actions,
                         torch.as_tensor(rewards, dtype=torch.float32),
                         values.detach(),
                         logp.detach(),
                         terminals)
            states = next_states
            self.total_steps += 1

        if (self.total_steps + 1) % 50000 == 0:

            if self.callback and self.callback.val_env:

                eval_score = self.callback.validate_policy(self.network['exploit'].state_dict(), device)
                if eval_score > self.best_eval_score:
                    self.save(self.config['save_dir'], self.total_steps, is_best=True)
                    self.best_eval_score = eval_score
                    with open(os.path.join(self.callback.policy_record.data_dir, 'best_eval_score.json'),
                              'w+') as f:
                        json.dump(self.best_eval_score, f)

                    # If it is best checkpoint so far, evaluate it
                    if self.callback.eval_env:
                        self.callback.evaluate_policy(self.network['exploit'].state_dict(), device)

        self.states = states
        with torch.no_grad():
            _, values, _, _ = network.step(torch.as_tensor(states, dtype=torch.float32).cuda())
        buffer.finish_path(values, env_idx=0)
        return buffer.get()

    def compute_kernel(self, states, actions):

        np_states = states.cpu().numpy()
        np_actions = torch.flatten(actions, end_dim=1).cpu().numpy()
        states_acts_cat = np.concatenate((np_states, np_actions), axis=1)
        phi = self.rbf_feature.transform(states_acts_cat)
        phi = torch.tensor(phi).cuda()
        return phi

    def update_density_model(self, mode):

        replay_buffer = self.replay_buffer[mode]
        replay_buffer_act = self.replay_buffer_actions[mode]
        states = torch.cat(sum(replay_buffer, []))
        actions = torch.cat(sum(replay_buffer_act, []))

        N = states.shape[0]
        ind = np.random.choice(N, min(2000, N), replace=False)
        with torch.no_grad():
            states_reshaped = torch.flatten(states, end_dim=1).cuda()
            states = self.network['exploit'].pi.cnn_net(states_reshaped)
        pdists = scipy.spatial.distance.pdist((states.cpu().numpy())[ind])
        self.rbf_feature.gamma = 1./(np.median(pdists)**2)
        phi = self.compute_kernel(states, actions)
        n, d = phi.shape
        sigma = torch.mm(phi.t(), phi) + self.config['ridge'] * torch.eye(d).cuda()
        self.density_model = torch.inverse(sigma).detach()

        covariance_matrices = []
        assert len(replay_buffer) == len(replay_buffer_act)
        for i in range(len(replay_buffer)):
            states = torch.cat(replay_buffer[i])
            actions = torch.cat(replay_buffer_act[i])
            with torch.no_grad():
                states_reshaped = torch.flatten(states, end_dim=1).cuda()
                states = self.network['exploit'].pi.cnn_net(states_reshaped)
            phi = self.compute_kernel(states, actions)
            n, d = phi.shape
            sigma = torch.mm(phi.t(), phi) + self.config['ridge'] * torch.eye(d).cuda()
            covariance_matrices.append(sigma.detach())
        m = 0
        for matrix in covariance_matrices:
            m = max(m, matrix.max())
        covariance_matrices = [matrix / m for matrix in covariance_matrices]

        if mode == 'explore-exploit':
            self.optimize_policy_mixture_weights(covariance_matrices)
        self.reward_bonus_normalizer = RescaleNormalizer()


    def optimize_policy_mixture_weights(self, covariance_matrices):
        d = covariance_matrices[0].shape[0]
        N = len(covariance_matrices)
        if N == 1:
            self.policy_mixture_weights = torch.tensor([1.0]).cuda()
        else:
            self.log_alphas = torch.nn.Parameter(torch.randn(N))
            opt = torch.optim.Adam([self.log_alphas], lr=0.001)
            for i in range(5000):
                opt.zero_grad()
                sigma_weighted_sum = torch.zeros(d, d).cuda()
                for n in range(N):
                    sigma_weighted_sum += F.softmax(self.log_alphas, dim=0)[n].cuda() * covariance_matrices[n]
                loss = -torch.logdet(sigma_weighted_sum)
                if torch.any(torch.isnan(loss)):
                    pdb.set_trace()
                loss.backward()
                opt.step()
            with torch.no_grad():
                self.policy_mixture_weights = F.softmax(self.log_alphas, dim=0).cuda()

    def update_replay_buffer(self):

        for mode in ['explore-exploit']:
            states, actions, returns, infos = [], [], [], []
            for _ in range(self.config['n_rollouts_for_density_est']):
                data = self.gather_trajectories(roll_in=False, add_bonus_reward=False, mode=mode,
                                                record_return=True)
                states += data['obs']
                returns += data['ret']
                actions += data['act']

            mean_return = torch.cat(returns).cpu().mean() * self.config['horizon']
            if mode == 'explore-exploit':
                self.policy_mixture_returns.append(mean_return.item())
            states = [s.cpu() for s in states]
            self.replay_buffer[mode].append(states)

            actions = [a.cpu() for a in actions]
            self.replay_buffer_actions[mode].append(actions)

    def optimize_policy(self):

        for mode in ['explore-exploit']:
            if mode == 'exploit' and self.epoch < self.config['start_exploit']:
                continue
            for i in range(self.config['n_policy_loops']):
                _ = self.step_optimize_policy(mode=mode)

        self.policy_mixture.append(copy.deepcopy(self.network['explore-exploit'].state_dict()))
        self.policy_mixture_optimizers.append(copy.deepcopy(self.optimizer['explore-exploit'].state_dict()))

    def initialize_new_policy(self, mode):

        self.network[mode] = core.ActorCritic(self.env.observation_space, self.env.action_space, **dict()).to(device)
        self.optimizer[mode] = Adam(self.network[mode].parameters(), lr=self.config['lr'])
        self.load_representation()

    def step_optimize_policy(self, mode):

        network = self.network[mode]
        optimizer = self.optimizer[mode]

        states, actions, rewards, log_probs_old, returns, advantages = [], [], [], [], [], []

        for i in range(self.config['n_traj_per_loop']):

            coin = np.random.rand()
            if coin <= (1.0-self.config['proll']):
                traj = self.gather_trajectories(add_bonus_reward=(mode=='explore-exploit'), mode=mode, roll_in=False,
                                                record_return=True)
            else:
                traj = self.gather_trajectories(add_bonus_reward=(mode=='explore-exploit'), mode=mode, roll_in=True,
                                                record_return=True)

            states += traj['obs']
            actions += traj['act']
            log_probs_old += traj['logp']
            returns += traj['ret']
            rewards += traj['rew']
            advantages += traj['adv']

        states = torch.cat(states, 0)
        actions = torch.cat(actions, 0)
        log_probs_old = torch.cat(log_probs_old, 0)
        returns = torch.cat(returns, 0)
        rewards = torch.cat(rewards, 0)
        advantages = torch.cat(advantages, 0)
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == advantages.shape[0] == returns.shape[0]

        actions = actions.detach()
        log_probs_old = log_probs_old.detach()

        for _ in range(self.config['optimization_epochs']):
            sampler = random_sample(np.arange(states.size(0)), self.config['mini_batch_size'])
            for batch_indices in sampler:
                batch_indices = torch.tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                pi, logp = network.pi(sampled_states, sampled_actions)
                ratio = (logp - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config['ppo_ratio_clip'],
                                          1.0 + self.config['ppo_ratio_clip']) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - self.config['entropy_weight'] * pi.entropy().mean()
                value_loss = 0.5 * (sampled_returns - network.v(sampled_states)).pow(2).mean()

                if torch.any(torch.isnan(policy_loss)) or torch.any(torch.isnan(value_loss)): pdb.set_trace()

                optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                optimizer.step()

        return rewards.mean()

    # extra helpers:

    def continuous_uniform_prob(self):
        act_max = np.asarray([1.0, 1.0, 1.0])
        act_min = np.asarray([-1.0, -1.0, -1.0])
        act_range = act_max - act_min
        prob = 1.
        for i in range(act_range.shape[0]):
            prob *= 1. / act_range[i]
        return prob

    def uniform_sample_cont_random_acts(self, N):
        acts = []
        for i in range(N):
            act = []
            for j in range(5):
                act.append([random.uniform(-1.0, 1.0),
                    random.uniform(-1.0, 1.0),
                    random.uniform(-1.0, 1.0)])
            acts.append(act)
        return np.array(acts)
