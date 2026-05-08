#!/usr/bin/env bash
# Task 3 spec submission run: full 6-trick stack, save model snapshots at
# env_step = 600k, 1M, 1.5M, 2M, 2.5M (spec requirement).
#
# Output: ./results/task3/snapshots/seed${SEED}/model_step{600000,...}.pt
#
# Episode budget: 2.5M env_steps at Pong's late-stage ~6k env_steps/episode
# is ~420 episodes; the EPISODES default (2000) from _pong_task3_base.sh is
# safe headroom. Override with EPISODES=N if you want a hard cap.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_pong_task3_base.sh"
cd "$SCRIPT_DIR/.."

CMD=(
    "$PYTHON_BIN" dqn.py
    "${BASE_ARGS[@]}"
    "${FULL_TRICKS[@]}"
    --wandb-run-name "pong-task3-snapshots-seed${SEED}"
    --save-dir "./results/task3/snapshots/seed${SEED}"
    --snapshot-env-steps 600000 1000000 1500000 2000000 2500000
    "$@"
)

printf '\n===== task3 snapshot run (seed=%s) =====\n' "$SEED"
printf '%q ' "${CMD[@]}"
printf '\n\n'

if [[ "$DRY_RUN" != "1" ]]; then
    "${CMD[@]}"
fi
