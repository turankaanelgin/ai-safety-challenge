import pdb

import numpy as np
import math
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_utils import *


def initialize_weights(mod, initialization_type, scale):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    '''
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")

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

    def sample(self, p):
        raise NotImplementedError

    def get_loglikelihood(self, p, actions):
        raise NotImplementedError

    def entropies(self, p):
        raise NotImplementedError

    def calc_kl(self, p, q, get_mean=True):
        raise NotImplementedError

    def forward(self, obs):
        raise NotImplementedError


class GaussianActor(Actor):

    def __init__(self, observation_space, act_dim, activation, cnn_net=None):
        super().__init__()
        log_std = np.log(0.5) * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.cnn_net = cnn_net

        dummy_img = torch.rand((1,) + observation_space.shape)
        self.mu_net = nn.Linear(cnn_net(dummy_img).shape[1], act_dim)
        initialize_weights(self.mu_net, 'orthogonal', scale=1.0)

    def forward(self, obs):

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

        if self.cnn_net is not None:
            batch_size = obs.shape[0]
            seq_size = obs.shape[1]
            obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
            obs = self.cnn_net(obs)
            obs = obs.reshape(batch_size, seq_size, obs.shape[1])

        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return mu, std

    def sample(self, p):

        mu, std = p
        return (mu + torch.randn_like(mu) * std).detach()

    def get_loglikelihood(self, p, actions):

        try:
            mean, std = p
            nll = 0.5 * ((actions - mean).pow(2) / std.pow(2)).sum(-1) \
                  + 0.5 * np.log(2.0 * np.pi) * actions.shape[-1] \
                  + self.log_std.sum(-1)
            return -nll
        except Exception:
            raise ValueError("Numerical error")

    def calc_kl(self, p, q, npg_approx=False, get_mean=True):

        p_mean, p_std = p
        q_mean, q_std = q
        p_var = p_std.pow(2) + 1e-10
        q_var = q_std.pow(2) + 1e-10

        d = q_mean.shape[1]
        logdetp = log_determinant(p_var)
        logdetq = log_determinant(q_var)
        diff = q_mean - p_mean

        log_quot_frac = logdetq - logdetp
        tr = (p_var / q_var).sum()
        quadratic = ((diff / q_var) * diff).sum(dim=1)

        if npg_approx:
            kl_sum = 0.5 * quadratic + 0.25 * (p_var / q_var - 1.).pow(2).sum()
        else:
            kl_sum = 0.5 * (log_quot_frac - d + tr + quadratic)
        assert kl_sum.shape == (p_mean.shape[0],)
        if get_mean:
            return kl_sum.mean()
        return kl_sum

    def entropies(self, p):

        _, std = p
        var = std.pow(2) + 1e-10
        logdetp = log_determinant(var)
        d = var.shape[0]
        entropies = 0.5 * (logdetp + d * (1. + math.log(2 * math.pi)))
        return entropies


class BetaActor(Actor):

    def __init__(self, observation_space, act_dim, activation, cnn_net=None):
        super().__init__()
        self.cnn_net = cnn_net
        self.action_space_high = 1.0
        self.action_space_low = -1.0
        self.action_dim = act_dim

        dummy_img = torch.rand((1,) + observation_space.shape)
        self.alpha_pre_softplus = nn.Linear(cnn_net(dummy_img).shape[1], act_dim)
        initialize_weights(self.alpha_pre_softplus, 'orthogonal', scale=0.01)
        self.beta_pre_softplus = nn.Linear(cnn_net(dummy_img).shape[1], act_dim)
        initialize_weights(self.beta_pre_softplus, 'orthogonal', scale=0.01)
        self.softplus = nn.Softplus()

    def forward(self, obs):

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

        if self.cnn_net is not None:
            batch_size = obs.shape[0]
            seq_size = obs.shape[1]
            obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
            obs = self.cnn_net(obs)
            obs = obs.reshape(batch_size, seq_size, obs.shape[1])

        alpha = torch.add(self.softplus(self.alpha_pre_softplus(obs)), 1.)
        beta = torch.add(self.softplus(self.beta_pre_softplus(obs)), 1.)
        return alpha, beta

    def scale_by_action_bounds(self, beta_dist_samples):
        # Scale [0, 1] back to action space.
        return beta_dist_samples * (self.action_space_high -
                                    self.action_space_low) + self.action_space_low

    def inv_scale_by_action_bounds(self, actions):
        # Scale action space to [0, 1].
        return (actions - self.action_space_low) / (self.action_space_high -
                                                    self.action_space_low)

    def sample(self, p):
        '''
        Given prob dist (alpha, beta), return: actions sampled from p_i, and their
        probabilities. p is tuple (alpha, beta). means shape
        (batch_size, action_space), var (action_space,), here are batch_size many
        prboability distributions you're sampling from
        Returns tuple (actions, probs):
        - actions: shape (batch_size, action_dim)
        - probs: shape (batch_size, action_dim)
        '''
        alpha, beta = p
        dist = torch.distributions.beta.Beta(alpha, beta)
        samples = dist.sample()
        return self.scale_by_action_bounds(samples)

    def get_loglikelihood(self, p, actions):
        alpha, beta = p
        dist = torch.distributions.beta.Beta(alpha, beta)
        log_probs = dist.log_prob(self.inv_scale_by_action_bounds(actions))
        return torch.sum(log_probs, dim=-1)

    def lbeta(self, alpha, beta):
        '''The log beta function.'''
        return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)

    def calc_kl(self, p, q, npg_approx=False, get_mean=True):
        '''
        Get the expected KL distance between beta distributions.
        '''
        assert not npg_approx
        p_alpha, p_beta = p
        q_alpha, q_beta = q

        # Expectation of log x under p.
        e_log_x = torch.digamma(p_alpha) - torch.digamma(p_alpha + p_beta)
        # Expectation of log (1-x) under p.
        e_log_1_m_x = torch.digamma(p_beta) - torch.digamma(p_alpha + p_beta)
        kl_per_action_dim = (p_alpha - q_alpha) * e_log_x
        kl_per_action_dim += (p_beta - q_beta) * e_log_1_m_x
        kl_per_action_dim -= self.lbeta(p_alpha, p_beta)
        kl_per_action_dim += self.lbeta(q_alpha, q_beta)
        # By chain rule on KL divergence.
        kl_joint = torch.sum(kl_per_action_dim, dim=1)
        if get_mean:
            return kl_joint.mean()
        return kl_joint

    def entropies(self, p):
        '''
        Get entropies over the probability distributions given by p
        p_i = (alpha, beta), p mean is shape (batch_size, action_space),
        p var is shape (action_space,)
        '''
        alpha, beta = p
        entropies = self.lbeta(alpha, beta)
        entropies -= (alpha - 1) * torch.digamma(alpha)
        entropies -= (beta - 1) * torch.digamma(beta)
        entropies += (alpha + beta - 2) * torch.digamma(alpha + beta)
        return torch.sum(entropies, dim=1)


class MLPCritic(nn.Module):

    def __init__(self, observation_space, activation, cnn_net=None):
        super().__init__()
        self.cnn_net = cnn_net
        if self.cnn_net:
            dummy_img = torch.rand((1,) + observation_space.shape)
            self.v_net = nn.Sequential(
                nn.Linear(cnn_net(dummy_img).shape[1], 1),
                activation()
            )

    def forward(self, obs):

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

        if self.cnn_net is not None:
            batch_size = obs.shape[0]
            seq_size = obs.shape[1]
            obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])

            obs = self.cnn_net(obs)
            obs = obs.reshape(batch_size, seq_size, obs.shape[1])

        v_out = self.v_net(obs).squeeze(0)
        return torch.squeeze(v_out, -1)


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, activation=nn.Tanh, use_beta=False,
                 use_rnn=False, use_popart=False, use_sde=False, local_std=False):
        super().__init__()
        cnn_net = cnn(observation_space)
        if use_beta:
            self.pi = BetaActor(observation_space, action_space.shape[0], activation,
                                cnn_net=cnn_net)
        else:
            self.pi = GaussianActor(observation_space, action_space.shape[0], activation,
                                    cnn_net=cnn_net)
        self.v = MLPCritic(observation_space, activation, cnn_net=cnn_net)

    def step(self, obs):
        with torch.no_grad():
            mean, std = self.pi(obs)
            a = self.pi.sample((mean, std))
            logp_a = self.pi.get_loglikelihood((mean, std), a)
            entropy = self.pi.entropies((mean, std))
            v = self.v(obs)
        return a, v, logp_a, entropy, mean, std

    def act(self, obs):
        return self.step(obs)[0]