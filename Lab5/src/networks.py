"""Q-network. One `DQN` class shared by CartPole (MLP) and Pong (Nature-CNN).

The architecture is selected from the input shape:
    - 1-D shape (e.g. (4,))      -> MLP
    - 3-D shape (C, H, W)         -> Nature-CNN [Mnih et al., 2015]

Optional dueling head [Wang et al., 2016]:
    Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))

Optional noisy linear layers [Fortunato et al., 2018]: replaces epsilon-greedy
exploration with parameterized noise injected into the head's linear weights.
Factorized Gaussian noise variant (Rainbow standard).
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise tensors are buffers (not learned) so they move with .to(device)
        # and are excluded from optimizer. Re-sampled by reset_noise().
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # Fortunato 2018 sec 3.2: sigma_0 / sqrt(p), p = in_features
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        # f(x) = sgn(x) * sqrt(|x|), used to factorize noise across in/out dims
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DQN(nn.Module):
    def __init__(
        self,
        num_actions: int,
        input_shape: Sequence[int] = (4,),
        hidden_sizes: Sequence[int] = (64, 64),
        dueling: bool = False,
        noisy: bool = False,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.input_shape = tuple(input_shape)
        self.dueling = dueling
        self.noisy = noisy
        self._linear_cls = NoisyLinear if noisy else nn.Linear

        if len(self.input_shape) == 1:
            self._mode = "mlp"
            self.features, head_in = self._build_mlp_features(self.input_shape[0], hidden_sizes)
        elif len(self.input_shape) == 3:
            self._mode = "cnn"
            self.features, head_in = self._build_cnn_features(self.input_shape)
        else:
            raise ValueError(
                f"Unsupported input_shape={self.input_shape}; "
                f"expected 1-D (vector) or 3-D (C,H,W)."
            )

        self._build_head(head_in, num_actions)

    # ---------- builders ----------
    @staticmethod
    def _build_mlp_features(in_dim: int, hidden_sizes: Sequence[int]) -> tuple[nn.Sequential, int]:
        # MLP body stays plain Linear; only the head gets NoisyLinear (matches
        # Fortunato 2018: noise on the final layers, body kept deterministic).
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        return nn.Sequential(*layers), prev

    @staticmethod
    def _build_cnn_features(input_shape: Sequence[int]) -> tuple[nn.Sequential, int]:
        c, _, _ = input_shape
        features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        return features, 64 * 7 * 7

    def _build_head(self, head_in: int, num_actions: int) -> None:
        L = self._linear_cls
        if self._mode == "cnn":
            stream = lambda out: nn.Sequential(L(head_in, 512), nn.ReLU(), L(512, out))
        else:
            stream = lambda out: L(head_in, out)

        if self.dueling:
            self.value_stream = stream(1)
            self.advantage_stream = stream(num_actions)
        else:
            self.fc = stream(num_actions)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mode == "cnn":
            x = x.float() / 255.0
        features = self.features(x)
        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        return self.fc(features)

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
