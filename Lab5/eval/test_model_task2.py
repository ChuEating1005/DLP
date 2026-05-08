"""Task 2 evaluator (ALE/Pong-v5, vanilla DQN with CNN).

Spec 4.2.2: 20 episodes, seeds 0-19, report mean reward.
Score percentage = (min(mean, 19) + 21) / 40 * 20%.

Usage:
    python test_model_task2.py --model-path LAB5_StudentID_task2.pt
"""
from __future__ import annotations

import argparse

import ale_py  # noqa: F401  registers ALE/* envs with gymnasium
import gymnasium as gym
import numpy as np
import torch

from src.networks import DQN
from src.preprocessor import Preprocessor


def evaluate(model_path: str, episodes: int = 20, frame_stack: int = 4) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(episodes))

    env = gym.make("ALE/Pong-v5", render_mode=None)
    num_actions = env.action_space.n
    pre = Preprocessor(frame_stack=frame_stack, crop=False)

    model = DQN(num_actions=num_actions, input_shape=(frame_stack, 84, 84)).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

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
    score_pct = (min(mean, 19.0) + 21.0) / 40.0 * 20.0
    print()
    print(f"  Episodes: {len(rewards)}")
    print(f"  Mean reward: {mean:+.2f}  (std={std:.2f}, min={min(rewards):+.1f}, max={max(rewards):+.1f})")
    print(f"  Task 2 score: {score_pct:.2f} / 20.00 %")
    return {"mean": mean, "std": std, "rewards": rewards, "score_pct": score_pct}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()
    evaluate(args.model_path, episodes=args.episodes)
