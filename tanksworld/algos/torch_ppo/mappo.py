import pdb
import numpy as np
import random
import torch
from torch.optim import Adam

from . import core
import os
import json
import math
import pickle
import cv2
import matplotlib.pyplot as plt

from tanksworld.minimap_util import *


device = torch.device('cuda')


class RolloutBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95,
                 n_rollout_threads=1, centralized=False):

        self.n_agents = 5
        self.obs_buf = torch.zeros(core.combined_shape_v3(size, n_rollout_threads, self.n_agents, obs_dim)).to(device)
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



class PPOPolicy():

    def __init__(self, env, callback, eval_mode=False, visual_mode=False, **kargs):
        self.kargs = kargs
        self.env = env
        self.callback = callback
        self.eval_mode = eval_mode
        self.visual_mode = visual_mode

    def run(self, num_steps):
        self.kargs.update({'steps_to_run': num_steps})
        if self.eval_mode:
            self.evaluate(episodes_to_run=num_steps, model_path=self.kargs['model_path'],
                          num_envs=self.kargs['n_envs'])
        elif self.visual_mode:
            self.visualize(episodes_to_run=num_steps, model_path=self.kargs['model_path'])
        else:
            self.learn(**self.kargs)


    def visualize(self, episodes_to_run, model_path, actor_critic=core.ActorCritic, ac_kwargs=dict()):

        episodes = 0
        observation = self.env.reset()

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
        ckpt = torch.load(model_path)
        self.ac_model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self.ac_model.eval()

        observation_list = []
        step = 0
        while episodes < episodes_to_run:

            with torch.no_grad():
                action, v, logp, _ = self.ac_model.step(torch.as_tensor(observation, dtype=torch.float32).to(device))
            next_observation, reward, done, info = self.env.step(action.cpu().numpy())
            overview = info[0]['overview'].astype(np.uint8)
            fig, axes = plt.subplots(1, 1)
            plt.imshow(overview)
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            observation_list.append(data)

            observation = next_observation

            if done[0]:
                env.reset()
                episodes += 1

            step += 1
            if step == 20:
                out = cv2.VideoWriter(
                    'out.mp4', cv2.VideoWriter_fourcc(*"MJPG"), 3, (640, 480), True
                )
                for img in observation_list:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    out.write(img)
                out.release()

                pdb.set_trace()


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


    def setup_model(self, actor_critic, pi_lr, vf_lr, ac_kwargs, enemy_model=None):

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.obs = self.env.reset()

        self.ac_model = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(device)
        self.pi_optimizer = Adam(self.ac_model.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac_model.v.parameters(), lr=vf_lr)

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
        if self.selfplay:
            ckpt_dict['enemy_model_state_dict'] = self.enemy_model.state_dict()
        torch.save(ckpt_dict, model_path)


    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data, clip_ratio):

        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        size = data['obs'].shape[0]
        obs = torch.flatten(obs, end_dim=1) if self.centralized else torch.flatten(obs, end_dim=2)
        act = torch.flatten(act, end_dim=1) if self.centralized else torch.flatten(act, end_dim=2)

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
        obs = torch.flatten(obs, end_dim=1) if self.centralized else torch.flatten(obs, end_dim=2)
        ret = torch.flatten(ret, end_dim=1) if self.centralized else torch.flatten(ret)

        values = self.ac_model.v(obs).squeeze(0)
        return ((values - ret) ** 2).mean()


    def compute_loss_entropy(self, data):
        logp = data['logp']
        return -torch.mean(-logp)


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
            self.loss_p_index += 1
            self.writer.add_scalar('loss/Policy_Loss', loss_pi, self.loss_p_index)
            if entropy_coef > 0.0:
                self.writer.add_scalar('loss/Entropy_Loss', loss_entropy, self.loss_p_index)
            #std = torch.exp(self.ac_model.pi.log_std)
            #self.writer.add_scalar('std_dev/move', std[0].item(), self.loss_p_index)
            #self.writer.add_scalar('std_dev/turn', std[1].item(), self.loss_p_index)
            #self.writer.add_scalar('std_dev/shoot', std[2].item(), self.loss_p_index)
            self.pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.loss_v_index += 1
            self.writer.add_scalar('loss/Value_Loss', loss_v, self.loss_v_index)
            self.vf_optimizer.step()



    def get_ally_heuristic_2(self, state_vector):

        # Orient against your closest ally to encourage exploration

        heuristic_actions = []
        for tank_idx in range(5):
            state = state_vector[tank_idx]
            x, y = state[0], -state[1]
            heading = state[2]
            min_distance = np.infty
            min_angle = 0

            all_ally_points = []
            for ally_idx in range(5):
                if ally_idx != tank_idx:
                    ally_state = state_vector[ally_idx]
                    ally_x, ally_y = ally_state[0], -ally_state[1]
                    dx = x - ally_x
                    dy = y - ally_y

                    rel_ally_x, rel_ally_y = point_relative_point_heading([ally_x, ally_y], [x, y], heading)
                    rel_ally_x = (rel_ally_x / UNITY_SZ) * SCALE + float(IMG_SZ) * 0.5
                    rel_ally_y = (rel_ally_y / UNITY_SZ) * SCALE + float(IMG_SZ) * 0.5
                    all_ally_points.append((rel_ally_x, rel_ally_y))

                    dist = math.sqrt(dx * dx + dy * dy)
                    angle = math.atan2(dy, dx)
                    if dist < min_distance:
                        min_distance = dist
                        min_angle = angle
                        if rel_ally_x < 64:
                            orient_coeff = 1
                        else:
                            orient_coeff = -1
                        if rel_ally_y > 64:
                            translate_coeff = 1
                        else:
                            translate_coeff = -1

            heuristic_action = np.asarray(([[translate_coeff * 0.5, orient_coeff * min_angle, -1.0]]))
            heuristic_actions.append(heuristic_action)

        return np.expand_dims(np.concatenate(heuristic_actions, axis=0), axis=0)


    def get_ally_heuristic(self, state_vector):

        # Orient towards your closest ally

        import matplotlib.pyplot as plt

        heuristic_actions = []
        for tank_idx in range(5):
            state = state_vector[tank_idx]
            x, y = state[0], -state[1]
            heading = state[2]
            min_distance = np.infty
            min_angle = 0

            all_ally_points = []
            for ally_idx in range(5):
                if ally_idx != tank_idx:
                    ally_state = state_vector[ally_idx]
                    ally_x, ally_y = ally_state[0], -ally_state[1]
                    dx = x - ally_x
                    dy = y - ally_y

                    rel_ally_x, rel_ally_y = point_relative_point_heading([ally_x, ally_y], [x, y], heading)
                    rel_ally_x = (rel_ally_x / UNITY_SZ) * SCALE + float(IMG_SZ) * 0.5
                    rel_ally_y = (rel_ally_y / UNITY_SZ) * SCALE + float(IMG_SZ) * 0.5
                    all_ally_points.append((rel_ally_x, rel_ally_y))

                    dist = math.sqrt(dx * dx + dy * dy)
                    angle = math.atan2(dy, dx)
                    if dist < min_distance:
                        min_distance = dist
                        min_angle = angle
                        if rel_ally_x < 64:
                            orient_coeff = -1
                        else:
                            orient_coeff = 1
                        if rel_ally_y > 64:
                            translate_coeff = -1
                        else:
                            translate_coeff = 1

            heuristic_action = np.asarray(([[translate_coeff * 0.5, orient_coeff * min_angle, -1.0]]))
            heuristic_actions.append(heuristic_action)

        return np.expand_dims(np.concatenate(heuristic_actions, axis=0), axis=0)


    def learn(self, actor_critic=core.ActorCritic, ac_kwargs=dict(), seed=-1,
              steps_per_epoch=800, steps_to_run=100000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
              vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97,
              target_kl=0.01, freeze_rep=True, entropy_coef=0.0, use_value_norm=False,
              tb_writer=None, selfplay=False, ally_heuristic=False, centralized=False,
              local_std=False, enemy_model=None, **kargs):

        env = self.env
        self.writer = tb_writer
        self.loss_p_index, self.loss_v_index = 0, 0
        self.set_random_seed(seed)
        ac_kwargs['init_log_std'] = kargs['init_log_std']
        ac_kwargs['centralized'] = centralized
        ac_kwargs['local_std'] = local_std

        self.centralized = centralized
        self.selfplay = selfplay

        print('POLICY SEED', seed)

        self.prev_ckpt = None
        self.setup_model(actor_critic, pi_lr, vf_lr, ac_kwargs, enemy_model=enemy_model)
        self.load_model(kargs['model_path'], kargs['cnn_model_path'], freeze_rep, steps_per_epoch)
        num_envs = kargs['n_envs']
        if self.callback:
            self.callback.init_model(self.ac_model)

        ep_ret = 0
        ep_len = 0
        ep_rb_dmg = np.zeros(num_envs)
        ep_br_dmg = np.zeros(num_envs)
        ep_rr_dmg = np.zeros(num_envs)

        buf = RolloutBuffer(self.obs_dim, self.act_dim, steps_per_epoch, gamma,
                            lam, n_rollout_threads=num_envs, centralized=centralized)

        if not os.path.exists(kargs['save_dir']):
            from pathlib import Path
            Path(kargs['save_dir']).mkdir(parents=True, exist_ok=True)

        step = self.start_step
        episode_lengths = []
        episode_returns = []
        episode_red_blue_damages, episode_red_red_damages, episode_blue_red_damages = [], [], []
        episode_stds = []
        # Damage for last hundred steps
        last_hundred_red_blue_damages = [[] for _ in range(num_envs)]
        last_hundred_red_red_damages = [[] for _ in range(num_envs)]
        last_hundred_blue_red_damages = [[] for _ in range(num_envs)]
        best_eval_score = self.best_eval_score
        if ally_heuristic:
            mixing_coeff = 0.8

        while step < steps_to_run:

            if (step + 1) % 50000 == 0 or step == 0: # Periodically save the model
                self.save_model(kargs['save_dir'], step)

            if (step + 1) % 100000 == 0 and selfplay: # Selfplay load enemy
                if self.prev_ckpt is not None:
                    self.enemy_model.load_state_dict(self.prev_ckpt)

            if (step + 1) % 50000 == 0: # Selfplay record prev checkpoint
                self.prev_ckpt = self.ac_model.state_dict()

            if (step + 1) % 25000 == 0 and ally_heuristic: # Heuristic anneal mixing coefficient
                if mixing_coeff >= 0.05:
                    mixing_coeff -= 0.05

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

            if ally_heuristic:
                heuristic_action = self.get_ally_heuristic_2(self.state_vector, obs)
                # Mix heuristic action with policy action
                coin = np.random.rand()
                if coin < mixing_coeff:
                    next_obs, r, terminal, info = env.step(heuristic_action)
                else:
                    next_obs, r, terminal, info = env.step(a.cpu().numpy())
            else:
                next_obs, r, terminal, info = env.step(a.cpu().numpy())

            if selfplay or enemy_model is not None:
                r = r[:, :5]
                a = ally_a
                obs = ally_obs

            ep_ret += np.average(np.sum(r, axis=1))
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
                        with torch.no_grad(): v[env_idx] = 0 if self.centralized else torch.zeros(5)
                    buf.finish_path(v, env_idx)

                if epoch_ended:
                    for env_idx in range(num_envs):
                        buf.finish_path(v, env_idx)

                episode_lengths.append(ep_len)
                episode_returns.append(ep_ret)
                episode_red_red_damages.append(ep_rr_dmg)
                episode_blue_red_damages.append(ep_br_dmg)
                episode_red_blue_damages.append(ep_rb_dmg)
                #std = torch.exp(self.ac_model.pi.log_std).cpu().detach().numpy()
                episode_stds.append([0,0,0])

                if epoch_ended:
                    self.update(buf, train_pi_iters, train_v_iters, target_kl, clip_ratio, entropy_coef)

                ep_ret = 0
                ep_len = 0
                ep_rb_dmg = np.zeros(num_envs)
                ep_br_dmg = np.zeros(num_envs)
                ep_rr_dmg = np.zeros(num_envs)

            if (step + 1) % 100 == 0:
                if self.callback:
                    self.callback.save_metrics_multienv(episode_returns, episode_lengths, episode_red_blue_damages,
                                                        episode_red_red_damages, episode_blue_red_damages,
                                                        episode_stds=episode_stds)

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

                if self.callback and self.callback.eval_env:
                    eval_score = self.callback.validate_policy(self.ac_model.state_dict(), device)
                    if eval_score > best_eval_score:
                        self.save_model(kargs['save_dir'], step, is_best=True)
                        best_eval_score = eval_score
                        with open(os.path.join(self.callback.policy_record.data_dir, 'best_eval_score.json'),
                                    'w+') as f:
                            json.dump(best_eval_score, f)
