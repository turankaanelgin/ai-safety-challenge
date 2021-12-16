"""Probability distributions."""
import pdb
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import torch as th
from gym import spaces
from torch import nn
from torch.distributions.normal import Normal


class Distribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self):
        super(Distribution, self).__init__()
        self.distribution = None

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """Create the layers and parameters that represent the distribution.
        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""

    @abstractmethod
    def proba_distribution(self, *args, **kwargs) -> "Distribution":
        """Set parameters of the distribution.
        :return: self
        """

    @abstractmethod
    def log_prob(self, x: th.Tensor) -> th.Tensor:
        """
        Returns the log likelihood
        :param x: the taken action
        :return: The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) -> Optional[th.Tensor]:
        """
        Returns Shannon's entropy of the probability
        :return: the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> th.Tensor:
        """
        Returns a sample from the probability distribution
        :return: the stochastic action
        """

    @abstractmethod
    def mode(self) -> th.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution
        :return: the stochastic action
        """

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution.
        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    @abstractmethod
    def actions_from_params(self, *args, **kwargs) -> th.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.
        :return: actions
        """

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.
        :return: actions and log prob
        """


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.
    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class TanhBijector(object):
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6):
        super(TanhBijector, self).__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: th.Tensor) -> th.Tensor:
        return th.tanh(x)

    @staticmethod
    def atanh(x: th.Tensor) -> th.Tensor:
        """
        Inverse of Tanh
        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: th.Tensor) -> th.Tensor:
        """
        Inverse tanh.
        :param y:
        :return:
        """
        eps = th.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: th.Tensor) -> th.Tensor:
        # Squash correction (from original SAC implementation)
        return th.log(1.0 - th.tanh(x) ** 2 + self.epsilon)


class StateDependentNoiseDistribution(Distribution):

    def __init__(
        self,
        action_dim: int,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        epsilon: float = 1e-6,
    ):
        super(StateDependentNoiseDistribution, self).__init__()
        self.action_dim = action_dim
        self.latent_sde_dim = None
        self.mean_actions = None
        self.log_std = None
        self._latent_sde = None
        self.full_std = full_std
        self.use_expln = use_expln
        self.squash_output = squash_output
        self.epsilon = epsilon
        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None
        if squash_output:
            self.bijector = TanhBijector(epsilon)
        else:
            self.bijector = None

    def get_std(self, log_std: th.Tensor) -> th.Tensor:

        if self.use_expln:
            below_threshold = th.exp(log_std) * (log_std <= 0)
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (th.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            std = th.exp(log_std)

        if self.full_std:
            return std
        return th.ones(self.latent_sde_dim, self.action_dim).to(log_std.device)

    def sample_weights(self, log_std: th.Tensor, batch_size: int=1) -> None:

        std = self.get_std(log_std)
        self.weights_dist = Normal(th.zeros_like(std), std)
        self.exploration_mat = self.weights_dist.rsample()
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def proba_distribution_net(
            self, latent_dim: int, log_std_init: float = -2.0, latent_sde_dim: Optional[int] = None
    ) -> Tuple[nn.Module, nn.Parameter]:

        mean_actions_net = nn.Linear(latent_dim, self.action_dim)
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        # Reduce the number of parameters if needed
        log_std = th.ones(self.latent_sde_dim, self.action_dim) if self.full_std else th.ones(self.latent_sde_dim, 1)
        # Transform it to a parameter so it can be optimized
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        # Sample an exploration matrix
        self.sample_weights(log_std)
        return mean_actions_net, log_std

    def proba_distribution(
        self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor
    ) -> "StateDependentNoiseDistribution":
        """
        Create the distribution given its parameters (mean, std)
        :param mean_actions:
        :param log_std:
        :param latent_sde:
        :return:
        """
        # Stop gradient if we don't want to influence the features
        self._latent_sde = latent_sde.detach()
        variance = th.mm(self._latent_sde ** 2, self.get_std(log_std) ** 2)
        self.distribution = Normal(mean_actions, th.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.bijector is not None:
            gaussian_actions = self.bijector.inverse(actions)
        else:
            gaussian_actions = actions
        # log likelihood for a gaussian
        log_prob = self.distribution.log_prob(gaussian_actions)
        # Sum along action dim
        log_prob = sum_independent_dims(log_prob)

        if self.bijector is not None:
            # Squash correction (from original SAC implementation)
            log_prob -= th.sum(self.bijector.log_prob_correction(gaussian_actions), dim=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        if self.bijector is not None:
            # No analytical form,
            # entropy needs to be estimated using -log_prob.mean()
            return None
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions

    def mode(self) -> th.Tensor:
        actions = self.distribution.mean
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions

    def get_noise(self, latent_sde: th.Tensor) -> th.Tensor:
        latent_sde = latent_sde.detach()
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return th.mm(latent_sde, self.exploration_mat)
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(1)
        # (batch_size, 1, n_actions)
        noise = th.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(1)

    def actions_from_params(
            self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std, latent_sde)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
            self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std, latent_sde)
        log_prob = self.log_prob(actions)
        return actions, log_prob