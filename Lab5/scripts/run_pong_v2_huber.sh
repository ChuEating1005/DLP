#!/usr/bin/env bash
# Task 2 ablation: v2-mnih-standard + Huber loss
# Difference vs run_pong_v2.sh:
#   --loss-fn-type mse -> huber  (with --huber-beta 1.0, paper-standard)
#   --lr 1e-4         -> 2.5e-4  (Huber clips |delta|>1 gradients to magnitude 1,
#                                  so we can take a larger step. 2.5e-4 matches
#                                  Mnih 2015 / Rainbow's Atari LR with Huber.)
# Everything else identical so the comparison isolates the loss-function change.
set -euo pipefail
cd "$(dirname "$0")/.."

python dqn.py \
    --env-name ALE/Pong-v5 \
    --wandb-project dlp-lab5-task2 \
    --wandb-run-name pong-v2-huber-lr2.5e-4 \
    --save-dir ./results/task2/v2-huber-lr2.5e-4/ \
    --episodes 10000 \
    --max-episode-steps 100000 \
    --batch-size 32 \
    --lr 2.5e-4 \
    --memory-size 100000 \
    --replay-start-size 10000 \
    --replay-buffer-type uniform \
    --target-update-frequency 1000 \
    --epsilon-scheduler linear \
    --epsilon-start 1.0 --epsilon-min 0.01 --epsilon-decay-steps 1000000 \
    --loss-fn-type huber \
    --huber-beta 1.0 \
    --train-per-step 1 \
    --train-frequency 4 \
    --frame-stack 4 \
    --eval-interval 20 \
    --checkpoint-interval 50 \
    "$@"
