import pdb, sys

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta

from .noisy import NoisyLinear


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


class RNDNetwork(nn.Module):

    def __init__(self):

        super(RNDNetwork, self).__init__()
        self.cnn_net = cnn(4)
        self.head = nn.Linear(9216, 16)

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
        batch_size, n_agents = obs.shape[0], obs.shape[1]
        obs = obs.reshape(batch_size * n_agents, obs.shape[2], obs.shape[3],
                          obs.shape[4])
        obs = self.cnn_net(obs)
        obs = obs.reshape(batch_size, n_agents, obs.shape[1])
        return self.head(obs)



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

    def __init__(self, observation_space, act_dim, activation, cnn_net, init_log_std=-0.5, local_std=False,
                 noisy=False):
        super().__init__()

        self.cnn_net = cnn_net

        dummy_img = torch.rand((1,) + observation_space.shape)
        if noisy:
            self.mu_net = NoisyLinear(cnn_net(dummy_img).shape[1], act_dim)
        else:
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
        try:
            return Normal(mu, std)
        except: pdb.set_trace()

    def _log_prob_from_distribution(self, pi, act):

        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution

    def resample(self):
        self.mu_net.resample()


class GaussianActor_v2(Actor):

    def __init__(self, observation_space, act_dim, activation, cnn_net, init_log_std=-0.5, local_std=False):
        super().__init__()

        self.cnn_net = cnn_net

        dummy_img = torch.rand((1,) + observation_space.shape)
        self.mu_net = nn.Sequential(
            nn.Linear(cnn_net(dummy_img).shape[1]*2, act_dim),
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

        modified_obs = []
        sum_obs = torch.sum(obs, dim=1, keepdims=True)
        for agent_idx in range(5):
            this_obs = obs[:,agent_idx,:].unsqueeze(1)
            other_obs = sum_obs - this_obs
            other_avg_obs = other_obs / 4
            modified_obs.append(torch.cat((this_obs, other_avg_obs), dim=-1))
        obs = torch.cat(modified_obs, dim=1)

        mu = self.mu_net(obs)
        if self.local_std:
            std = self.log_std_net(obs) - 0.5
            std = torch.exp(std)
        else:
            std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):

        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class SoftmaxActor(Actor):

    def __init__(self, observation_space, activation, cnn_net):
        super().__init__()

        self.cnn_net = cnn_net

        dummy_img = torch.rand((1,) + observation_space.shape)
        self.action = nn.Sequential(
            nn.Linear(cnn_net(dummy_img).shape[1], 50*50*2),
            activation(),
        )

    def _distribution(self, obs):

        obs = self.reshape_obs(obs)

        batch_size = obs.shape[0]
        seq_size = obs.shape[1]
        obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
        obs = self.cnn_net(obs)
        obs = obs.reshape(batch_size, seq_size, obs.shape[1])

        logits = self.action(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class CentralizedGaussianActor(Actor):

    def __init__(self, observation_space, act_dim, activation, cnn_net, init_log_std=-0.5, num_agents=5):
        super().__init__()
        log_std = init_log_std * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.cnn_net = cnn_net

        dummy_img = torch.rand((1,) + observation_space.shape)
        self.mu_net = nn.Sequential(
            nn.Linear(cnn_net(dummy_img).shape[1]*num_agents, act_dim*num_agents),
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

    def __init__(self, observation_space, activation, cnn_net, noisy=False):
        super().__init__()
        self.cnn_net = cnn_net

        dummy_img = torch.rand((1,) + observation_space.shape)
        if noisy:
            self.v_net = NoisyLinear(cnn_net(dummy_img).shape[1], 1)
        else:
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

    def resample(self):
        self.v_net.resample()


class CentralizedCritic(nn.Module):

    def __init__(self, observation_space, activation, cnn_net, num_agents=5):
        super().__init__()
        self.cnn_net = cnn_net

        dummy_img = torch.rand((1,) + observation_space.shape)
        self.v_net = nn.Sequential(
            nn.Linear(cnn_net(dummy_img).shape[1]*num_agents, num_agents),
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
                 use_beta=False, init_log_std=-0.5, centralized_critic=False,
                 centralized=False, local_std=False, discrete_action=False, noisy=False, num_agents=5):
        super().__init__()

        cnn_net = cnn(observation_space.shape[0])

        if centralized:
            self.pi = CentralizedGaussianActor(observation_space, action_space.shape[0], activation,
                                             cnn_net=cnn_net, init_log_std=init_log_std, num_agents=num_agents)
        elif use_beta:
            self.pi = BetaActor(observation_space, action_space.shape[0], cnn_net=cnn_net)
        elif discrete_action:
            self.pi = SoftmaxActor(observation_space, activation, cnn_net=cnn_net)
        else:
            self.pi = GaussianActor(observation_space, action_space.shape[0], activation, cnn_net=cnn_net,
                                    init_log_std=init_log_std, local_std=local_std, noisy=noisy)

        if centralized_critic:
            self.v = CentralizedCritic(observation_space, activation, cnn_net=cnn_net, num_agents=num_agents)
            #self.v = Critic(observation_space, activation, cnn_net=cnn_net, noisy=noisy)
        else:
            self.v = Critic(observation_space, activation, cnn_net=cnn_net, noisy=noisy)

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

    def resample(self):
        self.pi.resample()
        self.v.resample()

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class MLPCritic(nn.Module):

    def __init__(self, observation_space, activation, hidden_sizes=[64, 64], num_agents=5):
        super().__init__()
        obs_dim = observation_space.shape[-1]


        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [num_agents], activation)

    def forward(self, obs):
        v = self.v_net(obs)
#        v = torch.squeeze(v, -1) # Critical to ensure v has right shape.
#        if
        return v


class MLPCentralizedGaussianActor(Actor):

    def __init__(self, observation_space, act_dim, activation, init_log_std=-0.5, num_agents=5, hidden_sizes=[64,64]):
        super().__init__()

        log_std = init_log_std * np.zeros(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        obs_dim = observation_space.shape[-1]
        self.n_agents = num_agents
        self.act_dim = act_dim

        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim * num_agents], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
#        if self.n_agents > 1:
        mu = mu.reshape(-1, self.n_agents, self.act_dim)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        p = pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution
        return p

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, activation=nn.Tanh,
                 use_beta=False, init_log_std=-0.5, centralized_critic=False,
                 centralized=False, local_std=False, discrete_action=False, noisy=False, num_agents=5):
        super().__init__()


        self.pi = MLPCentralizedGaussianActor(observation_space, action_space.shape[0], activation,
                                             init_log_std=init_log_std, num_agents=num_agents)

        self.v = MLPCritic(observation_space, activation, num_agents=num_agents)

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

    def resample(self):
        self.pi.resample()
        self.v.resample()

