"""Q-network. One `DQN` class shared by CartPole (MLP) and Pong (Nature-CNN).

The architecture is selected from the input shape:
    - 1-D shape (e.g. (4,))      -> MLP
    - 3-D shape (C, H, W)         -> Nature-CNN [Mnih et al., 2015]
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DQN(nn.Module):
    """Deep Q-Network with shape-aware backbone.

    Args
    ----
    num_actions   : size of the discrete action space.
    input_shape   : tuple describing the *post-preprocessed* input.
                    (4,)         -> MLP (CartPole)
                    (4, 84, 84)  -> CNN (Pong)
    hidden_sizes  : MLP hidden layer widths (only used for MLP mode).
    """

    def __init__(
        self,
        num_actions: int,
        input_shape: Sequence[int] = (4,),
        hidden_sizes: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.input_shape = tuple(input_shape)

        if len(self.input_shape) == 1:
            self._mode = "mlp"
            self.network = self._build_mlp(self.input_shape[0], hidden_sizes, num_actions)
        elif len(self.input_shape) == 3:
            self._mode = "cnn"
            # ----- Task 2: CNN backbone (Nature-DQN) -----
            # Standard architecture so it works for Pong out of the box.
            self.network = self._build_cnn(self.input_shape, num_actions)
        else:
            raise ValueError(
                f"Unsupported input_shape={self.input_shape}; "
                f"expected 1-D (vector) or 3-D (C,H,W)."
            )

    # ---------- builders ----------
    @staticmethod
    def _build_mlp(in_dim: int, hidden_sizes: Sequence[int], num_actions: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_cnn(input_shape: Sequence[int], num_actions: int) -> nn.Sequential:
        c, h, w = input_shape
        # Asserts the canonical 84x84 used by Preprocessor.
        return nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mode == "cnn":
            # Inputs are uint8 frames stacked by the preprocessor; normalize to [0, 1].
            x = x.float() / 255.0
        return self.network(x)
