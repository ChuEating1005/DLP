#!/usr/bin/env bash
# Task 1: Vanilla DQN on CartPole-v1
set -euo pipefail
cd "$(dirname "$0")/.."

python dqn.py \
    --env-name CartPole-v1 \
    --wandb-project dlp-lab5 \
    --wandb-run-name task1-cartpole \
    --save-dir ./results/task1 \
    --episodes 2000 \
    --max-episode-steps 500 \
    --batch-size 64 \
    --lr 1e-3 \
    --memory-size 100000 \
    --replay-start-size 5000 \
    --replay-buffer-type uniform \
    --target-update-frequency 2000 \
    --epsilon-start 1.0 \
    --epsilon-scheduler exponential \
    --epsilon-decay 0.99995 \
    --epsilon-decay-steps 60000 \
    --epsilon-min 0.05 \
    --loss-fn-type huber \
    --huber-beta 2.5 \
    --eval-interval 10 \
    --eval-episodes 20 \
    --early-stop-reward 500 \
    --early-stop-patience 3 \
    --checkpoint-interval 100 \
    "$@"
