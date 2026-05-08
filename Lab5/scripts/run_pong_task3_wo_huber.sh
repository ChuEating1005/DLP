#!/usr/bin/env bash
# Leave-one-out: drop Huber (back to MSE). PER kept.
# NOTE: PER + MSE is unstable in theory (large TD errors -> huge loss -> grad
# explosion); included as ablation evidence, not as a recommended config.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_pong_task3_base.sh"
cd "$SCRIPT_DIR/.."

run_task3_ablation "wo_huber" \
    --replay-buffer-type prioritized \
    --per-alpha 0.6 --per-beta 0.4 --per-beta-increment 1e-6 \
    --loss-fn-type mse \
    --double-dqn \
    --n-step 3 \
    --dueling-dqn \
    --noisy-net \
    "$@"
