import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


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


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, observation_space, act_dim, hidden_sizes, activation, act_limit, cnn_net=None):
        super().__init__()
        self.cnn_net = cnn_net
        self.act_limit = act_limit
        if cnn_net is not None:
            dummy_img = torch.rand((1,) + observation_space.shape)
            self.mu_net = nn.Sequential(
                nn.Linear(cnn_net(dummy_img).shape[1], act_dim),
                activation()
            )
            self.log_std_net = nn.Sequential(
                nn.Linear(cnn_net(dummy_img).shape[1], act_dim),
                activation()
            )

    def forward(self, obs, deterministic=False, with_logprob=True):
        if self.cnn_net is not None:
            net_out = self.cnn_net(obs)
        mu = self.mu_net(net_out)
        log_std = self.log_std_net(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class MLPQFunction(nn.Module):

    def __init__(self, observation_space, act_dim, hidden_sizes, activation, cnn_net=None):
        super().__init__()
        self.cnn_net = cnn_net
        if self.cnn_net is not None:
            dummy_img = torch.rand((1,) + observation_space.shape)
            self.q_net = nn.Sequential(
                nn.Linear(cnn_net(dummy_img).shape[1]+act_dim, 1),
                activation()
            )

    def forward(self, obs, act):
        obs = self.cnn_net(obs)
        q = self.q_net(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, cnn_net=None):
        super().__init__()

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        cnn_net = cnn(observation_space)

        self.pi = SquashedGaussianMLPActor(observation_space, act_dim, hidden_sizes, activation, act_limit, cnn_net=cnn_net)
        self.q1 = MLPQFunction(observation_space, act_dim, hidden_sizes, activation, cnn_net=cnn_net)
        self.q2 = MLPQFunction(observation_space, act_dim, hidden_sizes, activation, cnn_net=cnn_net)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
