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
from torch.distributions.laplace import Laplace
from torch.distributions.bernoulli import Bernoulli



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def combined_shape_v2(length, seq_len, shape=None):
    if shape is None:
        return (length, seq_len,)
    return (length, seq_len, shape) if np.isscalar(shape) else (length, seq_len, *shape)

def cnn(observation_space):
    model = nn.Sequential(
        nn.Conv2d(observation_space.shape[0], 32, 8, 4),
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


class GaussianActor(Actor):

    def __init__(self, observation_space, act_dim, activation, cnn_net):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.cnn_net = cnn_net

        dummy_img = torch.rand((1,) + observation_space.shape)
        self.mu_net = nn.Sequential(
            nn.Linear(cnn_net(dummy_img).shape[1], act_dim),
            activation())

    def _distribution(self, obs):

        obs = self.reshape_obs(obs)

        batch_size = obs.shape[0]
        seq_size = obs.shape[1]
        obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
        obs = self.cnn_net(obs)
        obs = obs.reshape(batch_size, seq_size, obs.shape[1])

        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class Critic(nn.Module):

    def __init__(self, observation_space, activation, cnn_net=None):
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
        return torch.squeeze(v_out, -1) # Critical to ensure v has right shape.


class ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, activation=nn.Tanh, use_beta=False):
        super().__init__()

        cnn_net = cnn(observation_space)
        self.num_agents = 5

        if use_beta:
            self.pi = nn.ModuleList([BetaActor(observation_space, action_space.shape[0], cnn_net=cnn_net)] \
                                    * self.num_agents)
        else: # Default Gaussian
            self.pi = nn.ModuleList([GaussianActor(observation_space, action_space.shape[0], activation,
                                                   cnn_net=cnn_net)] * self.num_agents)

        self.v = nn.ModuleList([Critic(observation_space, activation, cnn_net=cnn_net)] * self.num_agents)

        self.use_beta = use_beta
        self.action_space_high = 1.0
        self.action_space_low = -1.0

    def scale_by_action_bounds(self, beta_dist_samples):
        # Scale [0, 1] back to action space.
        return beta_dist_samples * (self.action_space_high -
                                    self.action_space_low) + self.action_space_low

    def step(self, obs):

        with torch.no_grad():
            logp_a = []
            entropy = []
            action = []
            v = []

            for idx in range(self.num_agents):

                if len(obs.shape) == 5:
                    pi = self.pi[idx]._distribution(obs[:,idx])
                    v_ = self.v[idx](obs[:,idx])
                elif len(obs.shape) == 6:
                    pi = self.pi[idx]._distribution(obs[:,:,idx])
                    v_ = self.v[idx](obs[:,:,idx])

                a = pi.sample()
                logp_a_ = self.pi[idx]._log_prob_from_distribution(pi, a)
                entropy_ = pi.entropy()

                v.append(v_)
                logp_a.append(logp_a_)
                entropy.append(entropy_)
                action.append(a)

            action = torch.cat(action, dim=1)
            logp_a = torch.cat(logp_a, dim=1)
            entropy = torch.cat(entropy, dim=1)
            v = torch.cat(v, dim=1)

        return action, v, logp_a, entropy

    def act(self, obs):
        return self.step(obs)[0]