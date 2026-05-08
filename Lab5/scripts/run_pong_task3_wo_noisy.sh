#!/usr/bin/env bash
# Leave-one-out: drop NoisyNet (back to epsilon-greedy linear schedule).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_pong_task3_base.sh"
cd "$SCRIPT_DIR/.."

run_task3_ablation "wo_noisy" \
    --replay-buffer-type prioritized \
    --per-alpha 0.6 --per-beta 0.4 --per-beta-increment 1e-6 \
    --loss-fn-type huber --huber-beta 1.0 \
    --double-dqn \
    --n-step 3 \
    --dueling-dqn \
    --epsilon-scheduler linear \
    --epsilon-start 1.0 \
    --epsilon-min 0.01 \
    --epsilon-decay-steps 50000 \
    "$@"
