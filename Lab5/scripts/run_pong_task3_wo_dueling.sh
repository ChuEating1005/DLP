#!/usr/bin/env bash
# Leave-one-out: drop Dueling head (flat Q head).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_pong_task3_base.sh"
cd "$SCRIPT_DIR/.."

run_task3_ablation "wo_dueling" \
    --replay-buffer-type prioritized \
    --per-alpha 0.6 --per-beta 0.4 --per-beta-increment 1e-6 \
    --loss-fn-type huber --huber-beta 1.0 \
    --double-dqn \
    --n-step 3 \
    --noisy-net \
    "$@"
