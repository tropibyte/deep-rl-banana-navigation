"""Q-networks for the Banana Collector agent.

A single ``QNetwork`` class covers all four architectural variants used in the
ablation, chosen by two orthogonal flags:

* ``dueling``  -- split the head into value + advantage streams (Wang et al., 2016)
* ``noisy``    -- replace head Linear layers with NoisyLinear (Fortunato et al., 2018)
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Linear layer with factorised Gaussian noise on weights and bias.

    Replaces epsilon-greedy with learned, state-conditioned exploration: the
    network can *learn* to reduce sigma where it is already confident, instead
    of exploring uniformly at random everywhere forever.
    """

    def __init__(self, in_features: int, out_features: int, sigma_zero: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_zero = sigma_zero

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        # sigma_0 / sqrt(p) is the factorised-noise initialisation from the paper
        self.weight_sigma.data.fill_(self.sigma_zero * bound)
        self.bias_sigma.data.fill_(self.sigma_zero * bound)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        """Resample the factorised noise. Costs p+q normals instead of p*q."""
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Greedy evaluation uses the noise-free mean weights.
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


class QNetwork(nn.Module):
    """MLP action-value network with optional dueling head and noisy layers."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden: Sequence[int] = (128, 64),
        dueling: bool = False,
        noisy: bool = False,
    ):
        super().__init__()
        self.dueling = dueling
        self.noisy = noisy

        linear = (lambda i, o: NoisyLinear(i, o)) if noisy else (lambda i, o: nn.Linear(i, o))

        body: list[nn.Module] = []
        last = state_size
        for h in hidden[:-1]:
            body += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.body = nn.Sequential(*body)

        head_in, head_hidden = last, hidden[-1]
        if dueling:
            self.adv = nn.Sequential(linear(head_in, head_hidden), nn.ReLU(),
                                     linear(head_hidden, action_size))
            self.val = nn.Sequential(linear(head_in, head_hidden), nn.ReLU(),
                                     linear(head_hidden, 1))
        else:
            self.head = nn.Sequential(linear(head_in, head_hidden), nn.ReLU(),
                                      linear(head_hidden, action_size))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.body(state)
        if not self.dueling:
            return self.head(x)
        adv = self.adv(x)
        val = self.val(x)
        # Mean-centre the advantages so V and A are identifiable.
        return val + adv - adv.mean(dim=1, keepdim=True)

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
