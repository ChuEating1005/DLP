#!/usr/bin/env bash
# Task 2/3: DQN on ALE/Pong-v5
# For Task 3 (Enhanced), append: --replay-buffer-type prioritized --double-dqn --n-step 3
set -euo pipefail
cd "$(dirname "$0")/.."

python dqn.py \
    --env-name ALE/Pong-v5 \
    --wandb-project dlp-lab5-task2 \
    --wandb-run-name pong- \
    --save-dir ./results/task2/train-4step/ \
    --episodes 20000 \
    --max-episode-steps 108000 \
    --batch-size 32 \
    --lr 1e-4 \
    --memory-size 200000 \
    --replay-start-size 50000 \
    --replay-buffer-type uniform \
    --target-update-frequency 10000 \
    --epsilon-start 1.0 \
    --epsilon-decay 0.9999995 \
    --epsilon-min 0.05 \
    --loss-fn-type mse \
    --train-per-step 1 \
    --frame-stack 4 \
    --eval-interval 20 \
    --checkpoint-interval 50 \
    "$@"
