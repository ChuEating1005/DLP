"""Record one greedy-policy demo video for Task 1/2/3 checkpoints.

Outputs are written under demo_videos/ by default:
    task1_cartpole.mp4
    task2_pong_vanilla.mp4
    task3_pong_enhanced.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ale_py  # noqa: F401  register ALE envs
import cv2
import gymnasium as gym
import numpy as np
import torch

from src.networks import DQN
from src.preprocessor import Preprocessor


def detect_arch(state_dict: dict[str, torch.Tensor]) -> tuple[bool, bool]:
    keys = list(state_dict.keys())
    dueling = any("value_stream" in k or "advantage_stream" in k for k in keys)
    noisy = any("weight_sigma" in k or "bias_sigma" in k for k in keys)
    return dueling, noisy


def write_frame(writer: cv2.VideoWriter, frame: np.ndarray) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape (H,W,3), got {frame.shape}")
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def make_writer(path: Path, first_frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def load_model(
    model_path: Path,
    num_actions: int,
    input_shape: tuple[int, ...],
    device: torch.device,
    auto_arch: bool = False,
) -> DQN:
    state = torch.load(model_path, map_location=device, weights_only=True)
    dueling, noisy = detect_arch(state) if auto_arch else (False, False)
    model = DQN(num_actions=num_actions, input_shape=input_shape, dueling=dueling, noisy=noisy).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def greedy_action(model: DQN, state: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        t = torch.from_numpy(np.asarray(state)).unsqueeze(0).to(device)
        return int(model(t).argmax(dim=1).item())


def record_cartpole(model_path: Path, output_path: Path, seed: int, fps: float) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    model = load_model(model_path, env.action_space.n, (obs_dim,), device)

    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    first_frame = env.render()
    writer = make_writer(output_path, first_frame, fps)
    total = 0.0
    done = False
    try:
        write_frame(writer, first_frame)
        while not done:
            action = greedy_action(model, np.asarray(obs, dtype=np.float32), device)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
            write_frame(writer, env.render())
    finally:
        writer.release()
        env.close()
    return total


def record_pong(model_path: Path, output_path: Path, seed: int, fps: float, enhanced: bool) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    pre = Preprocessor(frame_stack=4, crop=enhanced)
    model = load_model(model_path, env.action_space.n, (4, 84, 84), device, auto_arch=enhanced)

    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    state = pre.reset(obs)
    first_frame = env.render()
    writer = make_writer(output_path, first_frame, fps)
    total = 0.0
    done = False
    try:
        write_frame(writer, first_frame)
        while not done:
            action = greedy_action(model, state, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
            state = pre.step(obs)
            write_frame(writer, env.render())
    finally:
        writer.release()
        env.close()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["task1", "task2", "task3", "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("demo_videos"))
    parser.add_argument("--task1-model", type=Path, default=Path("LAB5_111550093_task1.pt"))
    parser.add_argument("--task2-model", type=Path, default=Path("LAB5_111550093_task2.pt"))
    parser.add_argument("--task3-model", type=Path, default=Path("LAB5_111550093_task3_best.pt"))
    args = parser.parse_args()

    jobs: list[tuple[str, Path, callable[[], float]]] = []
    if args.task in {"task1", "all"}:
        out = args.output_dir / "task1_cartpole.mp4"
        jobs.append(("Task 1", out, lambda: record_cartpole(args.task1_model, out, args.seed, args.fps)))
    if args.task in {"task2", "all"}:
        out = args.output_dir / "task2_pong_vanilla.mp4"
        jobs.append(("Task 2", out, lambda: record_pong(args.task2_model, out, args.seed, args.fps, enhanced=False)))
    if args.task in {"task3", "all"}:
        out = args.output_dir / "task3_pong_enhanced.mp4"
        jobs.append(("Task 3", out, lambda: record_pong(args.task3_model, out, args.seed, args.fps, enhanced=True)))

    for name, out, fn in jobs:
        reward = fn()
        print(f"{name}: wrote {out}  reward={reward:+.1f}")


if __name__ == "__main__":
    main()
