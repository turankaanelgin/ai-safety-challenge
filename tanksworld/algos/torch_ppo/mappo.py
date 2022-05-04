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


device = torch.device('cuda')


class RolloutBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95,
                 n_rollout_threads=1, centralized=False, n_agents=5, use_state_vector=False, discrete_action=False):

        self.n_agents = n_agents
        if use_state_vector:
            self.obs_buf = torch.zeros(core.combined_shape_v2(size, n_rollout_threads, obs_dim)).to(device)
        else:
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
        self.use_state_vector = use_state_vector

    def store(self, obs, act, rew, val, logp, dones):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """

        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs if self.use_state_vector else obs.squeeze(2) 
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

        if self.eval_mode:
            self.evaluate(episodes_to_run=num_steps, model_path=self.kargs['model_path'],
                          num_envs=self.kargs['n_envs'], ac_kwargs=ac_kwargs)
        elif self.visual_mode:
            self.visualize(episodes_to_run=num_steps, model_path=self.kargs['model_path'], ac_kwargs=ac_kwargs)
        elif self.data_mode:
            self.collect_data(episodes_to_run=num_steps, model_path=self.kargs['model_path'], ac_kwargs=ac_kwargs)
        else:
            self.learn(**self.kargs)


    def collect_data(self, episodes_to_run, model_path, actor_critic=core.ActorCritic, ac_kwargs=dict()):

        episodes = 0
        observation = self.env.reset()

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
        ckpt = torch.load(model_path)
        self.ac_model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self.ac_model.eval()

        observations = None
        actions = None
        rewards = None
        next_observations = None
        file_cnt = 128

        while episodes < episodes_to_run:

            with torch.no_grad():
                action, v, logp, _ = self.ac_model.step(torch.as_tensor(observation, dtype=torch.float32).to(device))
            next_observation, reward, done, info = self.env.step(action.cpu().numpy())

            if observations is None:
                observations = observation.reshape(observation.shape[0]*observation.shape[1]*observation.shape[2],
                                                   observation.shape[3], observation.shape[4], observation.shape[5])
                actions = action.cpu().numpy().reshape(action.shape[0]*action.shape[1], action.shape[2])
                rewards = reward.flatten()
                next_observations = next_observation.reshape(next_observation.shape[0]*next_observation.shape[1]*\
                                                             next_observation.shape[2],
                                                             next_observation.shape[3], next_observation.shape[4],
                                                             next_observation.shape[5])
            else:
                observations = np.concatenate((observations,
                                              observation.reshape(
                                                  observation.shape[0] * observation.shape[1] * observation.shape[2],
                                                  observation.shape[3], observation.shape[4], observation.shape[5])
                                              ), axis=0)
                actions = np.concatenate((actions,
                                          action.cpu().numpy().reshape(action.shape[0]*action.shape[1], action.shape[2])), axis=0)
                rewards = np.concatenate((rewards, reward.flatten()), axis=0)
                next_observations = np.concatenate((next_observations,
                                                    next_observation.reshape(next_observation.shape[0]*next_observation.shape[1]*\
                                                             next_observation.shape[2],
                                                             next_observation.shape[3], next_observation.shape[4],
                                                             next_observation.shape[5])), axis=0)

            observation = next_observation

            if done[0]:
                observation = self.env.reset()
                episodes += 1

            if observations.shape[0] == 1000:
                with open('/scratch/dataset/data{}.pkl'.format(file_cnt), 'wb') as f:
                    pickle.dump((observations, actions, rewards, next_observations), f)

                observations = None
                actions = None
                rewards = None
                next_observations = None
                file_cnt += 1



    def visualize(self, episodes_to_run, model_path, actor_critic=core.ActorCritic, ac_kwargs=dict()):

        # Record the salient parts of video (the time windows with nonzero reward)

        matplotlib.use('agg')

        episodes = 0
        observation = self.env.reset()

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
        ckpt = torch.load(model_path)
        self.ac_model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self.ac_model.eval()

        overview_images = []
        rewards = []
        while episodes < episodes_to_run:

            with torch.no_grad():
                action, v, logp, _ = self.ac_model.step(torch.as_tensor(observation, dtype=torch.float32).to(device))
            next_observation, reward, done, info = self.env.step(action.cpu().numpy())
            overview = info[0]['overview'].astype(np.uint8)

            overview_images.append(overview)
            rewards.append(reward)

            observation = next_observation
            if done[0]:
                observation = self.env.reset()
                episodes += 1

        total_rewards = np.sum(np.concatenate(rewards, axis=0), axis=1)
        total_rewards = (total_rewards >= 0.1)
        bitmap = np.zeros(len(total_rewards),)
        for idx, nonzero in enumerate(total_rewards):
            if nonzero:
                bitmap[max(0,idx-5):idx+5] = 1

        overview_images = [overview_images[i] for i, bit in enumerate(bitmap) if bit == 1]
        observation_list = []
        for overview in overview_images:
            fig, axes = plt.subplots(1, 1)
            plt.imshow(overview)
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            observation_list.append(data)

        out = cv2.VideoWriter(
            os.path.join(self.callback.policy_record.data_dir, 'video.avi'),
            cv2.VideoWriter_fourcc(*"MJPG"), 3, (640, 480), True
        )
        for img in observation_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out.write(img)
        out.release()


    def evaluate(self, episodes_to_run, model_path, num_envs=10, actor_critic=core.ActorCritic, ac_kwargs=dict()):

        episodes = 0
        observation = self.env.reset()

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
        ckpt = torch.load(model_path)
        self.ac_model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self.ac_model.eval()

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

        while episodes < episodes_to_run:
            with torch.no_grad():
                action, v, logp, _ = self.ac_model.step(torch.as_tensor(observation, dtype=torch.float32).to(device))
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
                episodes += 1
                observation = self.env.reset()

                if episodes % 2 == 0 and episodes > 0:
                    avg_red_red_damages = np.mean(episode_red_red_damages)
                    avg_red_blue_damages = np.mean(episode_red_blue_damages)
                    avg_blue_red_damages = np.mean(episode_blue_red_damages)

                    with open(os.path.join(self.callback.policy_record.data_dir, 'mean_statistics.json'), 'w+') as f:
                        json.dump({'Number of games': episodes,
                                   'Red-Red-Damage': avg_red_red_damages.tolist(),
                                   'Red-Blue Damage': avg_red_blue_damages.tolist(),
                                   'Blue-Red Damage': avg_blue_red_damages.tolist()}, f, indent=4)

                    with open(os.path.join(self.callback.policy_record.data_dir, 'all_statistics.json'), 'w+') as f:
                        json.dump({'Number of games': episodes,
                                   'Red-Red-Damage': all_episode_red_red_damages,
                                   'Red-Blue Damage': all_episode_red_blue_damages,
                                   'Blue-Red Damage': all_episode_blue_red_damages}, f, indent=4)

                    avg_red_red_damages_per_env = np.mean(episode_red_red_damages, axis=0)
                    avg_red_blue_damages_per_env = np.mean(episode_red_blue_damages, axis=0)
                    avg_blue_red_damages_per_env = np.mean(episode_blue_red_damages, axis=0)

                    with open(os.path.join(self.callback.policy_record.data_dir, 'mean_statistics_per_env.json'),
                              'w+') as f:
                        json.dump({'Number of games': episodes,
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
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    def setup_model(self, actor_critic, pi_lr, vf_lr, ac_kwargs, enemy_model=None):

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.obs = self.env.reset()
        self.state_vector = np.zeros((12,6))

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
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
        size = data['obs'].shape[0]
        obs = torch.flatten(obs, end_dim=1)
        act = torch.flatten(act, end_dim=1)

        # Policy loss
        pi, logp = self.ac_model.pi(obs, act)
        logp = logp.reshape(size, -1, 1) if self.single_agent else logp.reshape(size, -1, 5)
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
        obs = torch.flatten(obs, end_dim=1) if self.centralized or self.centralized_critic\
                                            else torch.flatten(obs, end_dim=2)
        ret = torch.flatten(ret, end_dim=1) if self.centralized or self.centralized_critic\
                                            else torch.flatten(ret)
        values = self.ac_model.v(obs).squeeze(0)
        return ((values - ret) ** 2).mean()


    def compute_loss_entropy(self, data):
        logp = data['logp']
        return -torch.mean(-logp)


    def compute_loss_rnd(self, data):

        obs = data['obs']
        obs = torch.flatten(obs, end_dim=1) if self.centralized or self.centralized_critic \
                                            else torch.flatten(obs, end_dim=2)
        rnd_target = self.rnd_network(obs).detach()
        rnd_pred = self.rnd_pred_network(obs)
        rnd_loss = F.mse_loss(rnd_pred, rnd_target, reduction='none').mean()
        return rnd_loss


    def update(self, buf, train_pi_iters, train_v_iters, target_kl, clip_ratio, entropy_coef):

        data = buf.get()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data, clip_ratio)

            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                break

            if entropy_coef > 0.0:
                loss_entropy = self.compute_loss_entropy(data)
                loss = loss_pi + entropy_coef * loss_entropy
            else:
                loss = loss_pi

            loss.backward()

            if self.rnd:
                self.rnd_optimizer.zero_grad()
                loss_rnd = self.compute_loss_rnd(data)
                loss_rnd.backward()
                self.rnd_optimizer.step()

            self.loss_p_index += 1
            self.writer.add_scalar('loss/Policy_Loss', loss_pi, self.loss_p_index)
            if entropy_coef > 0.0:
                self.writer.add_scalar('loss/Entropy_Loss', loss_entropy, self.loss_p_index)
            std = torch.exp(self.ac_model.pi.log_std) if not self.discrete_action else torch.zeros((3))
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
              tb_writer=None, selfplay=False, ally_heuristic=False, enemy_heuristic=False, dense_reward=False,
              centralized=False, centralized_critic=False, local_std=False, enemy_model=None, single_agent=False,
              discrete_action=False, rnd=False, noisy=False, rnd_bonus=0.0, **kargs):

        env = self.env
        self.writer = tb_writer
        self.loss_p_index, self.loss_v_index = 0, 0
        self.set_random_seed(seed)

        ac_kwargs['init_log_std'] = kargs['init_log_std']
        ac_kwargs['centralized'] = centralized
        ac_kwargs['centralized_critic'] = centralized_critic
        ac_kwargs['local_std'] = local_std
        ac_kwargs['discrete_action'] = discrete_action
        ac_kwargs['noisy'] = noisy

        self.centralized = centralized
        self.centralized_critic = centralized_critic
        self.selfplay = selfplay
        self.single_agent = single_agent
        self.discrete_action = discrete_action
        self.rnd = rnd
        self.rnd_bonus = rnd_bonus
        self.use_state_vector = kargs['use_state_vector']

        if self.use_state_vector:
            actor_critic = core.MLPActorCritic

        print('POLICY SEED', seed)

        self.prev_ckpt = None
        self.setup_model(actor_critic, pi_lr, vf_lr, ac_kwargs, enemy_model=enemy_model)
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
                            lam, n_rollout_threads=num_envs, use_state_vector=self.use_state_vector, centralized=centralized,
                            #n_agents=1 if single_agent else 5,
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

        #fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        while step < steps_to_run:

            if noisy: self.ac_model.resample()

            if (step + 1) % 50000 == 0 or step == 0: # Periodically save the model
                self.save_model(kargs['save_dir'], step)

            if (step + 1) % 100000 == 0 and selfplay: # Selfplay load enemy
                if self.prev_ckpt is not None:
                    self.enemy_model.load_state_dict(self.prev_ckpt)

            if (step + 1) % 50000 == 0: # Selfplay record prev checkpoint
                self.prev_ckpt = self.ac_model.state_dict()

            if (step + 1) % 25000 == 0 and (ally_heuristic or enemy_heuristic): # Heuristic anneal mixing coefficient
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

                '''
                obs_to_disp = self.obs[0,0,0:3] * 255
                plt.imshow(obs_to_disp.astype(np.uint8).transpose(1,2,0))
                plt.savefig('./obs.png')
                next_obs, r, terminal, info = env.step(heuristic_action)
                obs_to_disp = next_obs[0, 0, 0:3] * 255
                plt.imshow(obs_to_disp.astype(np.uint8).transpose(1, 2, 0))
                plt.savefig('./next_obs.png')

                if (step+1) % 10 == 0:
                    pdb.set_trace()
                '''
            else:
                if discrete_action:
                    action1 = (a // 100) * 0.04 - 1
                    action2 = ((a % 100) // 2) * 0.04 - 1
                    action3 = (a % 100) % 2 - 0.5
                    action = torch.cat((action1.unsqueeze(-1), action2.unsqueeze(-1), action3.unsqueeze(-1)), dim=-1)
                    next_obs, r, terminal, info = env.step(action.cpu().numpy())
                else:
                    next_obs, r, terminal, info = env.step(a.cpu().numpy())
            extrinsic_reward = r.copy()

            if self.rnd:
                rnd_target = self.rnd_network(obs).detach()
                rnd_pred = self.rnd_pred_network(obs).detach()
                rnd_loss = F.mse_loss(rnd_pred, rnd_target, reduction='none').mean(2)
                intrinsic_reward = 1000*rnd_loss.detach().cpu().numpy()
                r += intrinsic_reward

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

            ep_ret += np.average(np.sum(extrinsic_reward, axis=1))
            if rnd:
                ep_intrinsic_ret += intrinsic_reward
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
                if rnd:
                    episode_intrinsic_returns.append(ep_intrinsic_ret)
                episode_red_red_damages.append(ep_rr_dmg)
                episode_blue_red_damages.append(ep_br_dmg)
                episode_red_blue_damages.append(ep_rb_dmg)
                std = torch.exp(self.ac_model.pi.log_std).cpu().detach().numpy() if not discrete_action else torch.zeros((3))
                episode_stds.append(std)

                if epoch_ended:
                    self.update(buf, train_pi_iters, train_v_iters, target_kl, clip_ratio, entropy_coef)

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

            if (step+1) % 50000 == 0:

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
