"""Task 3 evaluator (ALE/Pong-v5, enhanced DQN with Dueling + NoisyNet).

Spec 4.2.2: 20 episodes, seeds 0-19, report mean reward. Sample-efficiency
table maps environment-step milestone to score percentage when mean >= 19.

Auto-detects dueling/noisy from the saved state_dict keys so the same script
works for any of the 6 submitted snapshots (600k/1M/1.5M/2M/2.5M/best).

Usage:
    python test_model_task3.py --model-path LAB5_StudentID_task3_600000.pt
"""
from __future__ import annotations

import argparse

import ale_py  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

from src.networks import DQN
from src.preprocessor import Preprocessor


def detect_arch(state_dict: dict) -> tuple[bool, bool]:
    keys = list(state_dict.keys())
    dueling = any("value_stream" in k or "advantage_stream" in k for k in keys)
    noisy = any("weight_sigma" in k or "bias_sigma" in k for k in keys)
    return dueling, noisy


def evaluate(model_path: str, episodes: int = 20, frame_stack: int = 4) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(episodes))

    env = gym.make("ALE/Pong-v5", render_mode=None)
    num_actions = env.action_space.n
    pre = Preprocessor(frame_stack=frame_stack)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    dueling, noisy = detect_arch(state_dict)
    model = DQN(
        num_actions=num_actions,
        input_shape=(frame_stack, 84, 84),
        dueling=dueling,
        noisy=noisy,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Arch: dueling={dueling}  noisy={noisy}")

    rewards = []
    for ep, seed in enumerate(seeds):
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        s = pre.reset(obs)
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                t = torch.from_numpy(np.asarray(s)).unsqueeze(0).to(device)
                action = int(model(t).argmax(dim=1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            total += float(r)
            s = pre.step(obs)
            done = terminated or truncated
        rewards.append(total)
        print(f"  ep {ep:02d} (seed={seed}): reward={total:+.1f}")

    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    print()
    print(f"  Episodes: {len(rewards)}")
    print(f"  Mean reward: {mean:+.2f}  (std={std:.2f}, min={min(rewards):+.1f}, max={max(rewards):+.1f})")
    print(f"  Reaches score 19? {'YES' if mean >= 19.0 else 'no'}")
    return {"mean": mean, "std": std, "rewards": rewards, "dueling": dueling, "noisy": noisy}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()
    evaluate(args.model_path, episodes=args.episodes)
