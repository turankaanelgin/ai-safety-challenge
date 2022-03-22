import pdb

import numpy as np
import math
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def combined_shape_v2(length, seq_len, shape=None):
    if shape is None:
        return (length, seq_len,)
    return (length, seq_len, shape) if np.isscalar(shape) else (length, seq_len, *shape)

def combined_shape_v3(length, batch_len, seq_len, shape=None):
    if shape is None:
        return (length, batch_len, seq_len,)
    return (length, batch_len, seq_len, shape) if np.isscalar(shape) else (length, batch_len, seq_len, *shape)


def cnn(n_channels):
    model = nn.Sequential(
        nn.Conv2d(n_channels, 32, 8, 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU(),
        nn.Flatten()
    )
    return model


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def reshape_obs(self, obs):
        # Reshape observation for CNN
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == 6:
            if obs.shape[1] == 1:
                obs = obs.squeeze(1)
            elif obs.shape[2] == 1:
                obs = obs.squeeze(2)
            else:
                obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2],
                                  obs.shape[3], obs.shape[4], obs.shape[5])
        return obs

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class BetaActor(Actor):

    def __init__(self, observation_space, act_dim, cnn_net):
        super().__init__()
        self.cnn_net = cnn_net
        self.action_space_high = 1.0
        self.action_space_low = -1.0

        dummy_img = torch.rand((1,) + observation_space.shape)
        self.alpha_net = nn.Sequential(
            nn.Linear(cnn_net(dummy_img).shape[1], act_dim),
            nn.Softplus()
        )
        self.beta_net = nn.Sequential(
            nn.Linear(cnn_net(dummy_img).shape[1], act_dim),
            nn.Softplus()
        )

    def _distribution(self, obs):

        obs = self.reshape_obs(obs)

        batch_size, n_agents = obs.shape[0], obs.shape[1]
        obs = obs.reshape(batch_size*n_agents, obs.shape[2], obs.shape[3],
                          obs.shape[4])
        obs = self.cnn_net(obs)
        obs = obs.reshape(batch_size, n_agents, obs.shape[1])

        alpha = torch.add(self.alpha_net(obs), 1.)
        beta = torch.add(self.beta_net(obs), 1.)
        return Beta(alpha, beta)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            act = (act - self.action_space_low) / (self.action_space_high - self.action_space_low)
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class GaussianActor(Actor):

    def __init__(self, observation_space, act_dim, activation, cnn_net, init_log_std=-0.5, local_std=False):
        super().__init__()

        self.cnn_net = cnn_net

        dummy_img = torch.rand((1,) + observation_space.shape)
        self.mu_net = nn.Sequential(
            nn.Linear(cnn_net(dummy_img).shape[1], act_dim),
            activation()
        )

        self.local_std = local_std
        if local_std:
            self.log_std_net = nn.Sequential(
                nn.Linear(cnn_net(dummy_img).shape[1], act_dim),
                nn.Softplus()
            )
            self.offset = init_log_std
        else:
            log_std = init_log_std * np.ones(act_dim, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def _distribution(self, obs):

        obs = self.reshape_obs(obs)

        batch_size = obs.shape[0]
        seq_size = obs.shape[1]
        obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
        obs = self.cnn_net(obs)
        obs = obs.reshape(batch_size, seq_size, obs.shape[1])

        #obs = F.normalize(obs, dim=2)

        mu = self.mu_net(obs)
        if self.local_std:
            std = self.log_std_net(obs) - 0.5
            std = torch.exp(std)
        else:
            std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class CentralizedGaussianActor(Actor):

    def __init__(self, observation_space, act_dim, activation, cnn_net, init_log_std=-0.5):
        super().__init__()
        log_std = init_log_std * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.cnn_net = cnn_net

        dummy_img = torch.rand((1,) + observation_space.shape)
        self.mu_net = nn.Sequential(
            nn.Linear(cnn_net(dummy_img).shape[1]*5, act_dim*5),
            activation()
        )

    def _distribution(self, obs):

        obs = self.reshape_obs(obs)

        batch_size = obs.shape[0]
        n_agents = obs.shape[1]
        obs = obs.reshape(batch_size * n_agents, obs.shape[2], obs.shape[3], obs.shape[4])
        obs = self.cnn_net(obs)
        obs = obs.reshape(batch_size, n_agents*obs.shape[1])

        mu = self.mu_net(obs).reshape(-1, n_agents, 3)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution



class CentralizedGaussianActor_v2(Actor):

    def __init__(self, observation_space, act_dim, activation, cnn_net, init_log_std=-0.5):
        super().__init__()
        log_std = init_log_std * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.cnn_net = cnn_net

        dummy_img = [torch.rand((1,) + observation_space.shape)] * 5
        dummy_img = torch.cat(dummy_img, dim=1)
        self.mu_net = nn.Sequential(
            nn.Linear(cnn_net(dummy_img).shape[1], act_dim*5),
            activation()
        )

    def _distribution(self, obs):

        if len(obs.shape) == 5:
            obs = torch.flatten(obs, start_dim=1, end_dim=2)
        else:
            obs = torch.flatten(obs, start_dim=1, end_dim=3)

        obs = self.cnn_net(obs)

        mu = self.mu_net(obs).reshape(-1, 5, 3)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class Critic(nn.Module):

    def __init__(self, observation_space, activation, cnn_net):
        super().__init__()
        self.cnn_net = cnn_net

        dummy_img = torch.rand((1,) + observation_space.shape)
        self.v_net = nn.Sequential(
            nn.Linear(cnn_net(dummy_img).shape[1], 1),
            activation()
        )

    def reshape_obs(self, obs):
        # Reshape observation for CNN
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == 6:
            if obs.shape[1] == 1:
                obs = obs.squeeze(1)
            elif obs.shape[2] == 1:
                obs = obs.squeeze(2)
            else:
                obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2],
                                  obs.shape[3], obs.shape[4], obs.shape[5])
        return obs

    def forward(self, obs):

        obs = self.reshape_obs(obs)

        batch_size = obs.shape[0]
        seq_size = obs.shape[1]
        obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
        obs = self.cnn_net(obs)
        obs = obs.reshape(batch_size, seq_size, obs.shape[1])

        v_out = self.v_net(obs)
        return torch.squeeze(v_out, -1)  # Critical to ensure v has right shape.


class CentralizedCritic(nn.Module):

    def __init__(self, observation_space, activation, cnn_net):
        super().__init__()
        self.cnn_net = cnn_net

        dummy_img = torch.rand((1,) + observation_space.shape)
        self.v_net = nn.Sequential(
            nn.Linear(cnn_net(dummy_img).shape[1]*5, 5),
            activation()
        )

    def reshape_obs(self, obs):
        # Reshape observation for CNN
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == 6:
            if obs.shape[1] == 1:
                obs = obs.squeeze(1)
            elif obs.shape[2] == 1:
                obs = obs.squeeze(2)
            else:
                obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2],
                                  obs.shape[3], obs.shape[4], obs.shape[5])
        return obs

    def forward(self, obs):

        obs = self.reshape_obs(obs)

        batch_size = obs.shape[0]
        n_agents = obs.shape[1]
        obs = obs.reshape(batch_size * n_agents, obs.shape[2], obs.shape[3], obs.shape[4])
        obs = self.cnn_net(obs)
        obs = obs.reshape(batch_size, n_agents * obs.shape[1])

        v_out = self.v_net(obs)
        return v_out


class CentralizedCritic_v2(nn.Module):

    def __init__(self, observation_space, activation, cnn_net):
        super().__init__()
        self.cnn_net = cnn_net

        dummy_img = [torch.rand((1,) + observation_space.shape)] * 5
        dummy_img = torch.cat(dummy_img, dim=1)
        self.v_net = nn.Sequential(
            nn.Linear(cnn_net(dummy_img).shape[1], 5),
            activation()
        )

    def reshape_obs(self, obs):
        # Reshape observation for CNN
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == 6:
            if obs.shape[1] == 1:
                obs = obs.squeeze(1)
            elif obs.shape[2] == 1:
                obs = obs.squeeze(2)
            else:
                obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2],
                                  obs.shape[3], obs.shape[4], obs.shape[5])
        return obs

    def forward(self, obs):

        if len(obs.shape) == 5:
            obs = torch.flatten(obs, start_dim=1, end_dim=2)
        else:
            obs = torch.flatten(obs, start_dim=1, end_dim=3)

        obs = self.cnn_net(obs)

        v_out = self.v_net(obs)
        return v_out


class ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, activation=nn.Tanh,
                 use_beta=False, init_log_std=-0.5, centralized=False, local_std=False):
        super().__init__()

        cnn_net = cnn(observation_space.shape[0])

        if centralized:
            self.pi = CentralizedGaussianActor(observation_space, action_space.shape[0], activation,
                                             cnn_net=cnn_net, init_log_std=init_log_std)
        elif use_beta:
            self.pi = BetaActor(observation_space, action_space.shape[0], cnn_net=cnn_net)
        else:
            self.pi = GaussianActor(observation_space, action_space.shape[0], activation, cnn_net=cnn_net,
                                    init_log_std=init_log_std, local_std=local_std)

        if centralized:
            #self.v = CentralizedCritic(observation_space, activation, cnn_net=cnn_net)
            self.v = Critic(observation_space, activation, cnn_net=cnn_net)
        else:
            self.v = Critic(observation_space, activation, cnn_net=cnn_net)

        self.action_space_high = 1.0
        self.action_space_low = -1.0
        self.use_beta = use_beta

    def scale_by_action_bounds(self, beta_dist_samples):
        # Scale [0, 1] back to action space.
        return beta_dist_samples * (self.action_space_high -
                                    self.action_space_low) + self.action_space_low

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            if self.use_beta:
                a = self.scale_by_action_bounds(a)
            entropy = pi.entropy()
            v = self.v(obs)
        return a, v, logp_a, entropy

    def act(self, obs):
        return self.step(obs)[0]