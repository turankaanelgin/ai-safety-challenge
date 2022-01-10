import pdb

import numpy as np
import scipy.signal
import os

import torch
import torch.nn as nn


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

def cnn(observation_space):
    model = nn.Sequential(
        nn.Conv2d(observation_space.shape[0], 32, 8, 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2),
        nn.ReLU(),
        nn.AvgPool2d(4),
        nn.Flatten()
    )
    return model

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, cnn_net):
        super().__init__()
        #pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.cnn_net = cnn_net
        dummy_img = torch.rand((1,) + obs_dim)
        #self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.pi = nn.Sequential(
                    nn.Linear(cnn_net(dummy_img).shape[1], act_dim),
                    nn.Tanh(),
                )
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        obs = self.cnn_net(obs)
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, activation, cnn_net):
        super().__init__()
        self.cnn_net = cnn_net
        dummy_img = torch.rand((1,) + obs_dim)
        self.q = nn.Sequential(
                    nn.Linear(5*cnn_net(dummy_img).shape[1] + 5*act_dim, 1),
                    activation()
                )

    def forward(self, obs, act):
        all_inputs = [self.cnn_net(o) for o in obs] + [a for a in act]
        q = self.q(torch.cat(all_inputs, dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, common_actor, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        N = 5
        obs_dim = observation_space.shape
        act_dim = action_space.shape[0]
        act_limit = 1.0

        cnn_net = cnn(observation_space)

        # build policy and value functions
        if common_actor:
            pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, cnn_net)
            self.pis = [pi for i in range(N)]
            self.unique_pis = nn.ModuleList([pi])
        else:
            self.pis = []
            for i in range(N):
                pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, cnn_net)
                self.pis.append(pi)
            self.pis = nn.ModuleList(self.pis)
            self.unique_pis = self.pis

        self.q = MLPQFunction(obs_dim, act_dim, activation, cnn_net)

    def act(self, obs):
        actions = []
        for i in range(len(obs)):
            with torch.no_grad():
                a = self.pis[i](obs[i])
                actions.append(a.cpu().detach().numpy())
        return actions
