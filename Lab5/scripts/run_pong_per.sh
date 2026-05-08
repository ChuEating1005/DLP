#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python dqn.py \
    --env-name ALE/Pong-v5 \
    --wandb-project dlp-lab5-task2 \
    --wandb-run-name pong-task3-per \
    --save-dir ./results/task3/per/ \
    --episodes 10000 \
    --max-episode-steps 100000 \
    --batch-size 32 \
    --lr 1e-4 \
    --memory-size 100000 \
    --replay-start-size 10000 \
    --replay-buffer-type prioritized \
    --per-alpha 0.6 \
    --per-beta 0.4 \
    --per-beta-increment 1e-6 \
    --target-update-frequency 1000 \
    --epsilon-scheduler linear \
    --epsilon-start 1.0 --epsilon-min 0.01 --epsilon-decay-steps 1000000 \
    --loss-fn-type mse \
    --train-per-step 1 \
    --train-frequency 4 \
    --frame-stack 4 \
    --eval-interval 20 \
    --checkpoint-interval 50 \
    "$@"