"""Task 1 evaluator (CartPole-v1, vanilla DQN).

Spec 4.2.2: 20 episodes, seeds 0-19, report mean reward.
Score percentage = min(mean, 480) / 480 * 15%.

Usage:
    python test_model_task1.py --model-path LAB5_StudentID_task1.pt
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch

from src.networks import DQN


def evaluate(model_path: str, episodes: int = 20, seeds: list[int] | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seeds is None:
        seeds = list(range(episodes))

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = DQN(num_actions=num_actions, input_shape=(obs_dim,)).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    rewards = []
    for ep, seed in enumerate(seeds):
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
                action = int(model(t).argmax(dim=1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            total += float(r)
            done = terminated or truncated
        rewards.append(total)
        print(f"  ep {ep:02d} (seed={seed}): reward={total:+.1f}")

    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    score_pct = min(mean, 480.0) / 480.0 * 15.0
    print()
    print(f"  Episodes: {len(rewards)}")
    print(f"  Mean reward: {mean:+.2f}  (std={std:.2f}, min={min(rewards):+.1f}, max={max(rewards):+.1f})")
    print(f"  Task 1 score: {score_pct:.2f} / 15.00 %")
    return {"mean": mean, "std": std, "rewards": rewards, "score_pct": score_pct}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()
    evaluate(args.model_path, episodes=args.episodes)
