import pdb

import numpy as np
import math
import scipy.signal
from gym.spaces import Box, Discrete
#from arena5.core.utils import mpi_print

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli
from .distributions import StateDependentNoiseDistribution

from algos.torch_ppo import rnn


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

def combined_shape_v4(length, num_envs, num_agents, seq_len, shape=None):
    if shape is None:
        return (length, num_envs, num_agents, seq_len,)
    return (length, num_envs, num_agents, seq_len, shape) if np.isscalar(shape) \
        else (length, num_envs, num_agents, seq_len, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


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


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


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

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPGaussianActor(Actor):

    def __init__(self, observation_space, act_dim, hidden_sizes, activation, cnn_net=None, rnn_net=None, two_fc_layers=False,
                 local_std=False):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.cnn_net = cnn_net
        self.rnn_net = rnn_net

        if rnn_net is not None:
            self.mu_net = nn.Sequential(
                nn.Linear(512, act_dim),
                activation()
            )
            self.local_std = local_std
            if local_std:
                self.local_std_scale = nn.Linear(512, act_dim)

        elif cnn_net is not None:
            dummy_img = torch.rand((1,) + observation_space.shape)
            #mpi_print(dummy_img.shape)
            if two_fc_layers:
                self.mu_net = nn.Sequential(
                    nn.Linear(cnn_net(dummy_img).shape[1], 512),
                    activation(),
                    nn.Linear(512, act_dim),
                    activation()
                )
            else:
                self.mu_net = nn.Sequential(
                    nn.Linear(cnn_net(dummy_img).shape[1], act_dim),
                    activation()
                )

            self.local_std = local_std
            if local_std:
                self.local_std_scale = nn.Linear(cnn_net(dummy_img).shape[1], act_dim)

        else:
            obs_dim = observation_space.shape[0]
            self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):

        if len(obs.shape) == 4 and self.rnn_net is None:
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == 6 and self.rnn_net is None:
            if obs.shape[1] == 1:
                obs = obs.squeeze(1)
            elif obs.shape[2] == 1:
                obs = obs.squeeze(2)
            else:
                obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2],
                                  obs.shape[3], obs.shape[4], obs.shape[5])
        elif self.rnn_net is not None:
            if len(obs.shape) == 7 and obs.shape[3] == 1:
                obs = obs.squeeze(3)
            num_rollouts = obs.shape[0]
            num_agents = obs.shape[1]
            obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2],
                              obs.shape[3], obs.shape[4], obs.shape[5])

        if self.cnn_net is not None:
            batch_size = obs.shape[0]
            seq_size = obs.shape[1]
            obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
            obs = self.cnn_net(obs)
            obs = obs.reshape(batch_size, seq_size, obs.shape[1])
        if self.rnn_net is not None:
            hidden = self.rnn_net.init_hidden(batch_size)
            obs, _ = self.rnn_net(obs, hidden)
            obs = obs.reshape(num_rollouts, num_agents, -1)

        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        if self.local_std:
            local_scale = self.local_std_scale(obs)
            local_scale = torch.clamp(local_scale, 0.1, 1.0)
            std = local_scale * std

        if self.rnn_net is not None: mu = mu.unsqueeze(0)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, observation_space, hidden_sizes, activation, cnn_net=None, rnn_net=None,
                 use_popart=False, n_agents=5):
        super().__init__()
        self.cnn_net = cnn_net
        self.rnn_net = rnn_net

        if self.rnn_net is not None:
            self.v_net = nn.Sequential(
                nn.Linear(512, 1),
                activation()
            )

        elif self.cnn_net is not None:
            dummy_img = torch.rand((1,) + observation_space.shape)
            if not use_popart:
                self.v_net = nn.Sequential(
                    nn.Linear(cnn_net(dummy_img).shape[1]*n_agents, 1),
                    activation()
                )
            else:
                self.v_net = PopArt(cnn_net(dummy_img).shape[1], 1)
        else:
            obs_dim = observation_space.shape[0]
            self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def extract_features(self, obs):

        if len(obs.shape) == 4 and self.rnn_net is None:
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == 6 and self.rnn_net is None:
            if obs.shape[1] == 1:
                obs = obs.squeeze(1)
            elif obs.shape[2] == 1:
                obs = obs.squeeze(2)
            else:
                obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2],
                                  obs.shape[3], obs.shape[4], obs.shape[5])
        elif self.rnn_net is not None:
            if len(obs.shape) == 7 and obs.shape[3] == 1:
                obs = obs.squeeze(3)
            num_rollouts = obs.shape[0]
            num_agents = obs.shape[1]
            obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2],
                              obs.shape[3], obs.shape[4], obs.shape[5])

        if self.cnn_net is not None:
            batch_size = obs.shape[0]
            seq_size = obs.shape[1]
            obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
            obs = self.cnn_net(obs)
            obs = obs.reshape(batch_size, seq_size, obs.shape[1])
        if self.rnn_net is not None:
            hidden = self.rnn_net.init_hidden(batch_size)
            obs, _ = self.rnn_net(obs, hidden)
            obs = obs.reshape(num_rollouts, num_agents, -1)

        return obs

    def forward(self, obs):

        features = self.extract_features(obs)
        features = features.flatten(start_dim=1)
        v_out = self.v_net(features)
        if self.rnn_net is not None: v_out = v_out.unsqueeze(0)
        return torch.squeeze(v_out, -1) # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh, two_fc_layers=False,
                 use_rnn=False, use_popart=False, use_sde=False, use_beta=False, local_std=False):
        super().__init__()

        use_cnn = len(observation_space.shape) == 3
        cnn_net = None
        rnn_net = None
        if use_cnn:
            cnn_net = cnn(observation_space)
        if use_rnn:
            rnn_net = rnn.GRUNet(input_dim=9216, hidden_dim=1024, output_dim=512, n_layers=1)

        if isinstance(action_space, Box):
            if use_sde:
                self.pi = MLPSDEActor(observation_space, action_space.shape[0], hidden_sizes, activation,
                                      cnn_net=cnn_net)
            elif use_beta:
                self.pi = MLPBetaActor(observation_space, action_space.shape[0], cnn_net=cnn_net)
            else: # Default Gaussian
                self.pi = MLPGaussianActor(observation_space, action_space.shape[0], hidden_sizes, activation,
                                           cnn_net=cnn_net, rnn_net=rnn_net, two_fc_layers=two_fc_layers,
                                           local_std=local_std)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(observation_space, action_space.n, hidden_sizes, activation, cnn_net=cnn_net)

        # build value function
        self.v = MLPCritic(observation_space, hidden_sizes, activation, cnn_net=cnn_net, rnn_net=rnn_net,
                           use_popart=use_popart)
        self.use_sde = use_sde
        self.use_beta = use_beta
        self.action_space_high = 1.0
        self.action_space_low = -1.0

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
        if self.use_sde:
            a = a.reshape(a.shape[0]//5, 5, a.shape[1])
            entropy = entropy.reshape(entropy.shape[0]//5, 5, entropy.shape[1]) if not self.use_sde else entropy
            logp_a = logp_a.reshape(logp_a.shape[0]//5, 5)
        return a, v, logp_a, entropy

    def act(self, obs):
        return self.step(obs)[0]
