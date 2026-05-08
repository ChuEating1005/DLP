#!/usr/bin/env bash
# Shared base for Task 3 ablation scripts. Sourced (not executed) by
# run_pong_task3_{full,wo_*}.sh. Centralising BASE_ARGS prevents drift between
# the leave-one-out configs.
#
# Hyperparameters: identical to scripts/run_pong_task2.sh
# CLI: each ablation script accepts --seed N and forwards "$@" to dqn.py.

set -euo pipefail

PROJECT="${WANDB_PROJECT:-dlp-lab5-task2}"
SEED="${SEED:-42}"
EPISODES="${EPISODES:-2000}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

BASE_ARGS=(
    --env-name ALE/Pong-v5
    --wandb-project "$PROJECT"
    --episodes "$EPISODES"
    --max-episode-steps 100000
    --batch-size 32
    --lr 1e-4
    --memory-size 50000
    --replay-start-size 5000
    --discount-factor 0.99
    --target-update-frequency 500
    --train-per-step 4
    --train-frequency 4
    --frame-stack 4
    --eval-interval 20
    --eval-episodes 20
    --checkpoint-interval 50
    --seed "$SEED"
)

# Full 5-trick stack. wo_* scripts override one element by replacing the
# corresponding flag block in their own `EXTRA_ARGS`.
FULL_TRICKS=(
    --replay-buffer-type prioritized
    --per-alpha 0.6
    --per-beta 0.4
    --per-beta-increment 1e-6
    --loss-fn-type huber
    --huber-beta 1.0
    --double-dqn
    --n-step 3
    --dueling-dqn
    --noisy-net
)

run_task3_ablation() {
    local tag="$1"
    shift
    local extra_args=("$@")

    local cmd=(
        "$PYTHON_BIN" dqn.py
        "${BASE_ARGS[@]}"
        "${extra_args[@]}"
        --wandb-run-name "pong-task3-${tag}-seed${SEED}"
        --save-dir "./results/task3/ablation/${tag}/seed${SEED}"
    )

    printf '\n===== task3 ablation: %s (seed=%s) =====\n' "$tag" "$SEED"
    printf '%q ' "${cmd[@]}"
    printf '\n\n'

    if [[ "$DRY_RUN" != "1" ]]; then
        "${cmd[@]}"
    fi
}
