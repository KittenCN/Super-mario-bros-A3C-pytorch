"""Building blocks for Mario policy networks."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Linear layer with factorised Gaussian noise."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def forward(self, x):
        if self.training:
            self._sample_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def _sample_noise(self):
        epsilon_in = self._factorised_noise(self.in_features)
        epsilon_out = self._factorised_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _factorised_noise(size: int):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())


class ImpalaResidualBlock(nn.Module):
    """IMPALA residual block with two convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        residual = x
        x = self.block(x)
        return F.relu(x + residual, inplace=True)


class ImpalaBlock(nn.Module):
    """Convolutional block as described in IMPALA."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ImpalaResidualBlock(out_channels)
        self.res2 = ImpalaResidualBlock(out_channels)

    def forward(self, x):
        x = F.relu(self.conv(x), inplace=True)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


def compute_conv_output_size(input_shape: Tuple[int, int, int], base_channels: int, num_blocks: int) -> int:
    c = base_channels
    h, w = input_shape[1:]
    for _ in range(num_blocks):
        h = math.floor((h + 2 - 3) / 2 + 1)
        w = math.floor((w + 2 - 3) / 2 + 1)
        c *= 2
    return c * h * w


__all__ = ["ImpalaBlock", "ImpalaResidualBlock", "NoisyLinear", "compute_conv_output_size"]

