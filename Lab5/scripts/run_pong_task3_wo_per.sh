#!/usr/bin/env bash
# Leave-one-out: drop PER (uniform replay). Huber kept (loss is independent of buffer).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_pong_task3_base.sh"
cd "$SCRIPT_DIR/.."

run_task3_ablation "wo_per" \
    --replay-buffer-type uniform \
    --loss-fn-type huber --huber-beta 1.0 \
    --double-dqn \
    --n-step 3 \
    --dueling-dqn \
    --noisy-net \
    "$@"
