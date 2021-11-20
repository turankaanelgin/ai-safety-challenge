import pdb

import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
from arena5.core.utils import mpi_print

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli

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


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class CNNEncodeAttention(nn.Module):
    def __init__(self, observation_space):
        super(CNNEncodeAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(4, 4, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.cnn_encode = cnn(observation_space)

    def forward(self, observation):
        pooled = self.avg_pool(observation)
        attn = self.sigmoid(self.conv(pooled))
        input = attn * observation
        return self.cnn_encode(input)


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


class MLPCategoricalActor(Actor):

    def __init__(self, observation_space, act_dim, hidden_sizes, activation, cnn_net=None):
        super().__init__()
        self.cnn_net = cnn_net
        if cnn_net is not None:
            self.cnn_net = cnn_net
            dummy_img = torch.rand((1,) + observation_space.shape)
            #mpi_print(dummy_img.shape)
            self.logits_net = nn.Sequential(
                nn.Linear(cnn_net(dummy_img).shape[1], 512),
                activation(),
                nn.Linear(512, act_dim),
                activation())

        else:
            obs_dim = observation_space.shape[0]
            # self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
            self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        if self.cnn_net is not None:
            obs = self.cnn_net(obs)
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPBetaActor(Actor):

    def __init__(self, observation_space, act_dim, hidden_sizes, activation, cnn_net=None):
        super().__init__()
        if cnn_net is not None:
            self.cnn_net = cnn_net
            dummy_img = torch.rand((1,) + observation_space.shape)
            #mpi_print(dummy_img.shape)
            self.alpha_net = nn.Sequential(
                nn.Linear(cnn_net(dummy_img).shape[1], act_dim),
                activation()
            )
            self.beta_net = nn.Sequential(
                nn.Linear(cnn_net(dummy_img).shape[1], act_dim),
                activation()
            )

    def _distribution(self, obs):
        if self.cnn_net is not None:
            obs = self.cnn_net(obs)
        alpha = self.alpha_net(obs) + 1e-6
        beta = self.beta_net(obs) + 1e-6
        return Beta(alpha, beta)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPGaussianActor(Actor):

    def __init__(self, observation_space, act_dim, hidden_sizes, activation, cnn_net=None, rnn_net=None, two_fc_layers=False):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.cnn_net = cnn_net
        self.rnn_net = rnn_net

        if rnn_net is not None:
            self.mu_net = nn.Sequential(
                nn.Linear(1024, act_dim),
                activation()
            )

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
            from torchinfo import summary
            summary(self.cnn_net)
            summary(self.mu_net)

        else:
            obs_dim = observation_space.shape[0]
            self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == 6:
            if obs.shape[1] == 1:
                obs = obs.squeeze(1)
            else:
                obs = obs.squeeze(2)

        if self.cnn_net is not None:
            batch_size = obs.shape[0]
            seq_size = obs.shape[1]
            try:
                obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
            except:
                print('OBS', obs.shape)
                exit(0)
            obs = self.cnn_net(obs)
            obs = obs.reshape(batch_size, seq_size, obs.shape[1])
        if self.rnn_net is not None:
            hidden = self.rnn_net.init_hidden(batch_size)
            obs, _ = self.rnn_net(obs, hidden)
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, observation_space, hidden_sizes, activation, cnn_net=None, rnn_net=None):
        super().__init__()
        self.cnn_net = cnn_net
        self.rnn_net = rnn_net

        if self.rnn_net is not None:
            self.v_net = nn.Sequential(
                nn.Linear(1024, 1),
                activation()
            )

        elif self.cnn_net is not None:
            dummy_img = torch.rand((1,) + observation_space.shape)
            self.v_net = nn.Sequential(
                # nn.Linear(cnn_net(dummy_img).shape[1], 512),
                # activation(),
                # nn.Linear(512, 1),
                # activation()
                nn.Linear(cnn_net(dummy_img).shape[1], 1),
                activation()
            )
        else:
            obs_dim = observation_space.shape[0]
            self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        if len(obs.shape) == 4:
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == 6:
            obs = obs.squeeze(2)

        if self.cnn_net is not None:
            batch_size = obs.shape[0]
            seq_size = obs.shape[1]
            obs = obs.reshape(obs.shape[0]*obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
            obs = self.cnn_net(obs)
            obs = obs.reshape(batch_size, seq_size, obs.shape[1])
        if self.rnn_net is not None:
            hidden = self.rnn_net.init_hidden(batch_size)
            obs, _ = self.rnn_net(obs, hidden)
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh, two_fc_layers=False,
                 use_rnn=False):
        super().__init__()

        # policy builder depends on action space
        obs_dim = observation_space.shape[0]

        use_cnn = len(observation_space.shape) == 3
        cnn_net = None
        rnn_net = None
        if use_cnn:
            cnn_net = cnn(observation_space)
        if use_rnn:
            rnn_net = rnn.GRUNet(input_dim=9216, hidden_dim=2048, output_dim=1024, n_layers=2)

        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(observation_space, action_space.shape[0], hidden_sizes, activation,
                                       cnn_net=cnn_net, rnn_net=rnn_net, two_fc_layers=two_fc_layers)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(observation_space, action_space.n, hidden_sizes, activation, cnn_net=cnn_net)

        # build value function
        self.v = MLPCritic(observation_space, hidden_sizes, activation, cnn_net=cnn_net, rnn_net=rnn_net)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            entropy = pi.entropy()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a, v, logp_a, entropy

    def act(self, obs):
        return self.step(obs)[0]