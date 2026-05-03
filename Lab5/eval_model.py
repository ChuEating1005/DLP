"""Generic evaluator for Task 1 (CartPole) and Task 2/3 (Pong) snapshots.

Reuses the SAME DQN / Preprocessor classes used at training time, so the
input shape and preprocessing are guaranteed to match.

Per spec: 20 evaluation episodes with seeds 0..19, report mean reward.
"""
from __future__ import annotations

import argparse
import json
import os
from statistics import mean, pstdev

import numpy as np
import torch

from src.networks import DQN
from src.preprocessor import Preprocessor
from src.utils import infer_input_shape, is_atari_env, make_env, set_seed


def evaluate_one(env, model, preprocessor, device, max_steps: int) -> tuple[float, int]:
    obs, _ = env.reset()
    state = preprocessor.reset(obs)
    done = False
    total = 0.0
    steps = 0
    while not done and steps < max_steps:
        st = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            action = int(model(st).argmax().item())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
        state = preprocessor.step(next_obs)
        steps += 1
    return total, steps


def evaluate(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    atari = is_atari_env(args.env_name)
    env = make_env(args.env_name, render_mode="rgb_array")

    preprocessor = Preprocessor(
        mode="atari" if atari else "identity",
        frame_stack=args.frame_stack,
    )
    input_shape = infer_input_shape(env, frame_stack=args.frame_stack, atari=atari)
    num_actions = env.action_space.n

    model = DQN(num_actions, input_shape=input_shape).to(device)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    rewards: list[float] = []
    lengths: list[int] = []
    print(f"[Eval] model={args.model_path} env={args.env_name} "
          f"input_shape={input_shape} device={device}")

    for ep in range(args.episodes):
        seed = args.seed + ep
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
        r, n = evaluate_one(env, model, preprocessor, device, args.max_episode_steps)
        rewards.append(r)
        lengths.append(n)
        print(f"  ep {ep:02d} (seed={seed}): reward={r:.1f}  steps={n}")

    env.close()

    summary = {
        "model_path": args.model_path,
        "env_name": args.env_name,
        "episodes": args.episodes,
        "seeds": list(range(args.seed, args.seed + args.episodes)),
        "rewards": rewards,
        "mean": float(mean(rewards)),
        "std": float(pstdev(rewards)) if len(rewards) > 1 else 0.0,
        "min": float(min(rewards)),
        "max": float(max(rewards)),
        "mean_length": float(mean(lengths)),
    }
    print("\n=== Summary ===")
    print(f"  mean   : {summary['mean']:.2f}")
    print(f"  std    : {summary['std']:.2f}")
    print(f"  min/max: {summary['min']:.1f} / {summary['max']:.1f}")
    if args.env_name == "CartPole-v1":
        score_pct = min(summary["mean"], 480.0) / 480.0 * 15.0
        print(f"  Task 1 grade (15% cap): {score_pct:.2f}%")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--env-name", type=str, default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0,
                   help="First seed; episodes use seed, seed+1, ..., seed+N-1")
    p.add_argument("--max-episode-steps", type=int, default=10000)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--output-json", type=str, default="")
    args = p.parse_args()

    summary = evaluate(args)
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  saved -> {args.output_json}")


if __name__ == "__main__":
    main()
