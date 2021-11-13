import pdb
import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule


class MyActorCriticCnnPolicy(ActorCriticCnnPolicy):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(MyActorCriticCnnPolicy, self).__init__(
            observation_space, action_space, lr_schedule, net_arch, activation_fn, ortho_init,
            use_sde, log_std_init, full_std, sde_net_arch, use_expln, squash_output,
            features_extractor_class, features_extractor_kwargs, normalize_images,
            optimizer_class, optimizer_kwargs
        )

        cnn_ckpt = './models/frozen-cnn-0.8/4000000.pth'
        state_dict = th.load(cnn_ckpt)

        temp_state_dict = {}
        for key in state_dict:
            if 'cnn_net' in key:
                temp_state_dict['cnn.{}'.format(key.split('cnn_net.')[1])] = state_dict[key]

        self.features_extractor.load_state_dict(temp_state_dict, strict=True)

        for name, param in self.features_extractor.named_parameters():
            param.requires_grad = False

    def extract_features(self, obs: th.Tensor) -> th.Tensor:

        assert self.features_extractor is not None, "No features extractor was set"
        return self.features_extractor(obs)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = actions.reshape(actions.shape[0]*actions.shape[1], actions.shape[2])
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()