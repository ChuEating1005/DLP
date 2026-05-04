"""Replay buffers.

Spec-required class:
    - PrioritizedReplayBuffer

We also expose a thin `UniformReplayBuffer` that has the *same* sample API
(returning indices and IS-weights of 1) so the agent code path is uniform.

Both buffers return:
    states, actions, rewards, next_states, dones, indices, weights

Task 3 (PER) `add`, `sample`, `update_priorities` are scaffolded with TODO
markers and clear docstrings for later implementation.
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool] # (state, action, reward, next_state, done)

# --------------------------------------------------------------------------
# Uniform buffer (Task 1 / 2)
# --------------------------------------------------------------------------
class UniformReplayBuffer:
    """Vanilla FIFO replay buffer with uniform sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: list[Transition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition, error: float | None = None) -> None:  # noqa: ARG002
        # `error` accepted for API symmetry with PER; ignored here.
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            return [], [], [], [], [], [], np.zeros((0,), dtype=np.float32)
        batch_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        weights = np.ones((batch_size,), dtype=np.float32)
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, errors) -> None:  # noqa: ARG002
        # No-op for uniform sampling; kept for API parity with PER.
        return


# --------------------------------------------------------------------------
# Prioritized buffer (Task 3) -- name kept per spec
# --------------------------------------------------------------------------
class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay (Schaul et al., 2016)."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 2.5e-6,
        eps: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.buffer: list[Transition] = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        # Hyperparameters
        self.alpha = alpha # prioritization exponent (0 = uniform, 1 = full prioritization)
        self.beta = beta # importance-sampling exponent (0 = no correction, 1 = full correction)
        self.beta_increment = beta_increment # amount to increment beta after each sampling step (anneal towards 1)
        self.eps = eps # small constant to avoid zero priority        

    def __len__(self) -> int:
        return len(self.buffer)

    # ----------------------------------------------------------------------
    def add(self, transition: Transition, error: float | None = None) -> None:
        """Insert a transition with priority based on its TD error.

        New transitions get max priority (so they are sampled at least once).

        TODO (Task 3):
          1. Compute priority p = (|error| + eps) ** alpha if error given,
             else use max(self.priorities) (or 1.0 if buffer is empty).
          2. Append/overwrite at self.pos (circular).
          3. Store priority into self.priorities[self.pos].
          4. Advance self.pos modulo capacity.
        """
        if error is None:
            p = self.priorities.max() if len(self.buffer) > 0 else 1.0
        else:
            p = (abs(error) + self.eps) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        # Update priority
        self.priorities[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity
        return

    # ----------------------------------------------------------------------
    def sample(self, batch_size: int):
        """Sample a batch with probability ∝ p_i, return IS-weights. """
        N = len(self.buffer)
        probs = self.priorities[:N] / self.priorities[:N].sum()
        indices = np.random.choice(N, batch_size, p=probs)
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max() # Normalized
        self.beta = min(1.0, self.beta + self.beta_increment)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones, indices, weights

    # ----------------------------------------------------------------------
    def update_priorities(self, indices, errors) -> None:
        """Update priorities of sampled transitions after training."""
        for idx, err in zip(indices, errors):
            self.priorities[idx] = (abs(err) + self.eps) ** self.alpha
        return


# --------------------------------------------------------------------------
def make_replay_buffer(kind: str, capacity: int, **kwargs):
    """Factory used by DQNAgent."""
    kind = kind.lower()
    if kind == "uniform":
        return UniformReplayBuffer(capacity)
    if kind in ("prioritized", "per"):
        return PrioritizedReplayBuffer(
            capacity,
            alpha=kwargs.get("alpha", 0.6),
            beta=kwargs.get("beta", 0.4),
            beta_increment=kwargs.get("beta_increment", 1e-6),
            eps=kwargs.get("eps", 1e-6),
        )
    raise ValueError(f"Unknown replay buffer kind: {kind!r}")
