import torch
import torch.nn as nn
from .core import cnn


class ForwardBackwardModel(nn.Module):

    def __init__(self):
        super(ForwardBackwardModel, self).__init__()

        self.cnn = cnn(4)
        self.forward_model = ForwardModel(self.cnn)
        self.backward_model = BackwardModel(self.cnn)

    def forward(self, obs, action, next_obs):
        next_pred, next_feat = self.forward_model(obs, action, next_obs)
        action_pred = self.backward_model(obs, next_obs)
        return next_pred, next_feat, action_pred


class ForwardModel(nn.Module):

    def __init__(self, cnn_model):
        super(ForwardModel, self).__init__()

        self.cnn = cnn_model
        self.fc1 = nn.Linear(9219, 9216, bias=True)

    def forward(self, obs, action, next_obs):
        curr_feature = self.cnn(obs)
        next_feature = self.cnn(next_obs)
        return self.fc1(torch.cat((curr_feature, action), dim=1)), next_feature


class BackwardModel(nn.Module):

    def __init__(self, cnn_model):
        super(BackwardModel, self).__init__()

        self.cnn = cnn_model
        self.fc1 = nn.Linear(9216*2, 3, bias=True)

    def forward(self, obs, next_obs):
        curr_feature = self.cnn(obs)
        next_feature = self.cnn(next_obs)
        return self.fc1(torch.cat((curr_feature, next_feature), dim=1))
