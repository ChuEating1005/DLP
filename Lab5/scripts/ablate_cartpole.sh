#!/usr/bin/env bash
# Task 1 ablations for CartPole-v1.
# Runs one-factor-at-a-time experiments around a conservative baseline.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${WANDB_PROJECT:-dlp-lab5}"
SEEDS="${SEEDS:-42}"
EPISODES="${EPISODES:-1500}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

BASE_ARGS=(
    --env-name CartPole-v1
    --wandb-project "$PROJECT"
    --episodes "$EPISODES"
    --max-episode-steps 500
    --batch-size 64
    --lr 1e-3
    --memory-size 100000
    --replay-start-size 5000
    --replay-buffer-type uniform
    --target-update-frequency 1000
    --epsilon-start 1.0
    --epsilon-scheduler exponential
    --epsilon-decay 0.99995
    --epsilon-decay-steps 60000
    --epsilon-min 0.05
    --loss-fn-type huber
    --huber-beta 1.0
    --eval-interval 10
    --eval-episodes 20
    --early-stop-reward 480
    --early-stop-patience 3
    --checkpoint-interval 100
)

run_exp() {
    local name="$1"
    local seed="$2"
    shift 2

    local cmd=(
        "$PYTHON_BIN" dqn.py
        "${BASE_ARGS[@]}"
        --seed "$seed"
        --wandb-run-name "CartPole-${name}-seed${seed}"
        --save-dir "./results/ablation/${name}/seed${seed}"
        "$@"
    )

    printf '\n===== %s seed=%s =====\n' "$name" "$seed"
    printf '%q ' "${cmd[@]}"
    printf '\n'

    if [[ "$DRY_RUN" != "1" ]]; then
        "${cmd[@]}"
    fi
}

for seed in $SEEDS; do
    run_exp "baseline" "$seed"

    # Learning rate: high LR can inflate Q estimates quickly.
    run_exp "lr-2.5e-3" "$seed" --lr 2.5e-3
    run_exp "lr-5e-3" "$seed" --lr 5e-3

    # Exploration decay: slower decay keeps the replay buffer more diverse.
    run_exp "eps-fast-0.9999" "$seed" --epsilon-decay 0.9999
    run_exp "eps-slow-0.99998" "$seed" --epsilon-decay 0.99998
    run_exp "eps-linear-60k" "$seed" --epsilon-scheduler linear --epsilon-decay-steps 60000
    run_exp "eps-cosine-60k" "$seed" --epsilon-scheduler cosine --epsilon-decay-steps 60000
    run_exp "eps-linear-120k" "$seed" --epsilon-scheduler linear --epsilon-decay-steps 120000

    # Replay warmup: too little warmup can train on a narrow early distribution.
    run_exp "warmup-1000" "$seed" --replay-start-size 1000
    run_exp "warmup-10000" "$seed" --replay-start-size 10000

    # Target network cadence: less frequent syncs often reduce moving-target noise.
    run_exp "target-500" "$seed" --target-update-frequency 500
    run_exp "target-2000" "$seed" --target-update-frequency 2000

    # Huber beta: smaller beta is more robust to outliers but can underfit large errors.
    run_exp "huber-beta-2.5" "$seed" --huber-beta 2.5
    run_exp "huber-beta-5.0" "$seed" --huber-beta 5.0

    # Batch and loss sensitivity.
    # run_exp "batch-128" "$seed" --batch-size 128
    # run_exp "loss-mse" "$seed" --loss-fn-type mse
done
