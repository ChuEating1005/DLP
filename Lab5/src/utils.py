"""Generic helpers: seeding, env factory, tensor conversion, weight init."""
from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py

# Register Atari envs once (idempotent).
gym.register_envs(ale_py)


# ---------- tensor / init helpers ----------

def to_tensor(x, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Numpy/list -> torch.Tensor on `device` with given dtype."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device=device, dtype=dtype)


def init_weights(m: nn.Module) -> None:
    """Kaiming-uniform init for Conv2d / Linear layers."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ---------- reproducibility ----------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- env factory ----------

def make_env(env_name: str, render_mode: str = "rgb_array", seed: int | None = None) -> gym.Env:
    """Build a gym env. Works for CartPole-v1 and ALE/Pong-v5 (and others)."""
    env = gym.make(env_name, render_mode=render_mode)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
    return env


def is_atari_env(env_name: str) -> bool:
    """Heuristic: ALE prefix or '-v5' suffix typical for Atari."""
    return env_name.startswith("ALE/") or env_name.lower().startswith("atari")


# ---------- shape helper ----------

def infer_input_shape(env: gym.Env, frame_stack: int = 4, atari: bool = False) -> Tuple[int, ...]:
    """Return the *post-preprocess* input shape fed to DQN.

    - Atari (after Preprocessor): (frame_stack, 84, 84)
    - Vector envs: env.observation_space.shape (e.g. CartPole -> (4,))
    """
    if atari:
        return (frame_stack, 84, 84)
    return tuple(env.observation_space.shape)


# ---------- file utilities ----------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
