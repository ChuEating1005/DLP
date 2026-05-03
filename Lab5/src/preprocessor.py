"""Observation preprocessor.

A single class `Preprocessor` (name kept per spec) supporting two modes:
    - mode='atari'    -> grayscale + resize 84x84 + frame stacking
    - mode='identity' -> pass-through (for vector envs like CartPole)

Both modes share the same API so DQNAgent code stays identical:
    state = preprocessor.reset(obs)
    state = preprocessor.step(obs)
"""
from __future__ import annotations

from collections import deque
from typing import Literal

import cv2
import numpy as np

Mode = Literal["atari", "identity"]


class Preprocessor:
    def __init__(self, mode: Mode = "atari", frame_stack: int = 4) -> None:
        if mode not in ("atari", "identity"):
            raise ValueError(f"mode must be 'atari' or 'identity', got {mode!r}")
        self.mode = mode
        self.frame_stack = frame_stack
        self.frames: deque = deque(maxlen=frame_stack)

    # ---------- atari helpers ----------
    @staticmethod
    def _to_gray_84(obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    # ---------- public API ----------
    def reset(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "identity":
            return np.asarray(obs, dtype=np.float32)
        frame = self._to_gray_84(obs)
        self.frames = deque([frame.copy() for _ in range(self.frame_stack)],
                            maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)  # (C, 84, 84) uint8

    def step(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "identity":
            return np.asarray(obs, dtype=np.float32)
        frame = self._to_gray_84(obs)
        self.frames.append(frame.copy())
        return np.stack(self.frames, axis=0)
