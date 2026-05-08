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
    def __init__(self, mode: Mode = "atari", frame_stack: int = 4, crop: bool = True) -> None:
        if mode not in ("atari", "identity"):
            raise ValueError(f"mode must be 'atari' or 'identity', got {mode!r}")
        self.mode = mode
        self.frame_stack = frame_stack
        self.crop = crop
        self.frames: deque = deque(maxlen=frame_stack)

    # ---------- atari helpers ----------
    @staticmethod
    def _to_gray(obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        return gray

    @staticmethod
    def _resize(obs: np.ndarray) -> np.ndarray:
        return cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _to_play_region(gray: np.ndarray) -> np.ndarray:
        # Crop to play region (removing score, borders, etc.)
        return gray[34:34+160, :]

    # ---------- public API ----------
    def reset(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "identity":
            return np.asarray(obs, dtype=np.float32)
        frame = self._to_gray(obs)
        if self.crop:
            frame = self._to_play_region(frame)
        frame = self._resize(frame)
        self.frames = deque([frame.copy() for _ in range(self.frame_stack)],
                            maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)  # (C, 84, 84) uint8

    def step(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "identity":
            return np.asarray(obs, dtype=np.float32)
        frame = self._to_gray(obs)
        if self.crop:
            frame = self._to_play_region(frame)
        frame = self._resize(frame)
        self.frames.append(frame.copy())
        return np.stack(self.frames, axis=0)
