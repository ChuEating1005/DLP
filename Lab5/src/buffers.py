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

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


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
    """Proportional Prioritized Experience Replay (Schaul et al., 2016).

    Stored priorities follow p_i = (|delta_i| + eps) ** alpha (raised to alpha
    on insert) so sampling reduces to a categorical over priorities directly.

    NOTE: Task 3 -- to be filled in later.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-6,
        eps: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps

        self.buffer: list[Transition] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

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
        ########## YOUR CODE HERE (for Task 3) ##########

        ########## END OF YOUR CODE (for Task 3) ##########
        return

    # ----------------------------------------------------------------------
    def sample(self, batch_size: int):
        """Sample a batch with probability ∝ p_i, return IS-weights.

        TODO (Task 3):
          1. probs = self.priorities[:len(self)] / sum
          2. indices = np.random.choice(len(self), batch_size, p=probs)
          3. weights = (len(self) * probs[indices]) ** (-self.beta)
             weights /= weights.max()       # normalize
          4. self.beta = min(1.0, self.beta + self.beta_increment)
          5. return states, actions, rewards, next_states, dones, indices, weights
        """
        ########## YOUR CODE HERE (for Task 3) ##########

        ########## END OF YOUR CODE (for Task 3) ##########
        return

    # ----------------------------------------------------------------------
    def update_priorities(self, indices, errors) -> None:
        """Update priorities of sampled transitions after training.

        TODO (Task 3):
          For each (idx, err) pair:
              self.priorities[idx] = (|err| + eps) ** alpha
        """
        ########## YOUR CODE HERE (for Task 3) ##########

        ########## END OF YOUR CODE (for Task 3) ##########
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
