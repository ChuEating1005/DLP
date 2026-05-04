"""Entry script for DLP Lab5 - DQN training.

Same script for CartPole-v1 (Task 1) and ALE/Pong-v5 (Task 2/3).
The agent autodetects the env type and picks MLP/CNN + atari/identity preprocessor.

Examples:
    python dqn.py --env-name CartPole-v1   --wandb-run-name task1-cartpole
    python dqn.py --env-name ALE/Pong-v5   --wandb-run-name task2-pong \
        --memory-size 200000 --replay-start-size 50000 --batch-size 32 \
        --episodes 5000 --max-episode-steps 108000
"""
from __future__ import annotations

import argparse

import wandb

from src.agent import DQNAgent
from src.utils import set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--env-name", type=str, default="CartPole-v1",
                   help="CartPole-v1 | ALE/Pong-v5 | ...")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="./results")
    p.add_argument("--wandb-project", type=str, default="DLP-Lab5-DQN")
    p.add_argument("--wandb-run-name", type=str, default="run")

    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max-episode-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--discount-factor", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--loss-fn-type", type=str, default="mse", choices=["mse", "huber"])
    p.add_argument("--huber-beta", type=float, default=1.0)
    p.add_argument("--train-per-step", type=int, default=1)
    p.add_argument("--train-frequency", type=int, default=4,
                   help="Number of environment steps between training updates.")

    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-scheduler", type=str, default="exponential",
                   choices=["exponential", "linear", "cosine", "constant"],
                   help="Epsilon schedule applied per environment step.")
    p.add_argument("--epsilon-decay", type=float, default=0.999999)
    p.add_argument("--epsilon-decay-steps", type=int, default=60_000,
                   help="Number of env steps to anneal epsilon for linear/cosine schedules.")
    p.add_argument("--epsilon-min", type=float, default=0.05)

    p.add_argument("--memory-size", type=int, default=100_000)
    p.add_argument("--replay-start-size", type=int, default=50_000)
    p.add_argument("--replay-buffer-type", type=str, default="uniform",
                   choices=["uniform", "prioritized"])

    # PER hyperparameters
    p.add_argument("--per-alpha", type=float, default=0.6,
                   help="(PER) prioritization exponent (0 = uniform, 1 = full).")
    p.add_argument("--per-beta", type=float, default=0.4,
                   help="(PER) initial IS-weight exponent; annealed toward 1.0.")
    p.add_argument("--per-beta-increment", type=float, default=1e-6,
                   help="(PER) per-sample increment to anneal beta toward 1.0.")

    p.add_argument("--target-update-frequency", type=int, default=1000)
    p.add_argument("--frame-stack", type=int, default=4)

    # Double DQN
    p.add_argument("--double-dqn", action="store_true",
                   help="(Task 3) Enable Double DQN target.")
    
    # Multi-Step Return
    p.add_argument("--n-step", type=int, default=1,
                   help="(Task 3) n-step return (1 = vanilla).")

    p.add_argument("--eval-interval", type=int, default=20,
                   help="Run a greedy eval episode every N training episodes.")
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--eval-seed", type=int, default=10_000,
                   help="First seed used for deterministic evaluation episodes.")
    p.add_argument("--early-stop-reward", type=float, default=None,
                   help="Stop once eval reward consistently reaches this threshold.")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Number of consecutive evals that must reach early-stop reward; 0 stops on the first passing eval.")
    p.add_argument("--checkpoint-interval", type=int, default=100)

    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        save_code=True,
        config=vars(args),
    )

    # Define custom metrics with step tracking for better wandb visualization
    wandb.define_metric("global/env_step")
    wandb.define_metric("global/update_step")
    wandb.define_metric("rollout/*", step_metric="global/env_step")
    wandb.define_metric("eval/*", step_metric="global/env_step")
    wandb.define_metric("train/*", step_metric="global/update_step")

    agent = DQNAgent(env_name=args.env_name, args=args)
    try:
        agent.run(episodes=args.episodes)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
