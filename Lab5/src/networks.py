"""Q-network. One `DQN` class shared by CartPole (MLP) and Pong (Nature-CNN).

The architecture is selected from the input shape:
    - 1-D shape (e.g. (4,))      -> MLP
    - 3-D shape (C, H, W)         -> Nature-CNN [Mnih et al., 2015]

Optional dueling head [Wang et al., 2016]:
    Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(
        self,
        num_actions: int,
        input_shape: Sequence[int] = (4,),
        hidden_sizes: Sequence[int] = (64, 64),
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.input_shape = tuple(input_shape)
        self.dueling = dueling

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
        hidden = 512 if self._mode == "cnn" else head_in
        if self._mode == "cnn":
            stream = lambda out: nn.Sequential(
                nn.Linear(head_in, hidden), nn.ReLU(), nn.Linear(hidden, out)
            )
        else:
            stream = lambda out: nn.Linear(head_in, out)

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
        else:
            return self.fc(features)
