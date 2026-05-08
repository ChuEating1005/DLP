#!/usr/bin/env bash
# Task 3 full 5-trick run: DDQN + PER + n-step=3 + Dueling + NoisyNet.
# Identical to scripts/run_pong_task3.sh; lives here for ablation symmetry
# (lets you launch full + 6 wo_* on 7 GPUs with the same script pattern).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_pong_task3_base.sh"
cd "$SCRIPT_DIR/.."

run_task3_ablation "full" "${FULL_TRICKS[@]}" "$@"
