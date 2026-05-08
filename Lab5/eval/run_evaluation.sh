#!/usr/bin/env bash
# Reproduce all reported evaluation results for Lab 5.
#
# Spec 4.2.2 evaluation protocol: 20 episodes, seeds 0..19, mean reward.
# Run from the repo root inside the same Python env as requirements.txt.
#
# Usage:
#     bash run_evaluation.sh           # run all tasks
#     bash run_evaluation.sh task1     # run a single task
#     bash run_evaluation.sh task2
#     bash run_evaluation.sh task3
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python}
SID=111550093

run_task1() {
    echo "============================================================"
    echo " Task 1 (CartPole-v1, vanilla DQN)"
    echo "============================================================"
    $PY test_model_task1.py --model-path "LAB5_${SID}_task1.pt"
}

run_task2() {
    echo "============================================================"
    echo " Task 2 (ALE/Pong-v5, vanilla DQN with CNN)"
    echo "============================================================"
    $PY test_model_task2.py --model-path "LAB5_${SID}_task2.pt"
}

run_task3() {
    echo "============================================================"
    echo " Task 3 (ALE/Pong-v5, enhanced DQN: Dueling + NoisyNet + DDQN + PER + n-step)"
    echo "============================================================"
    for milestone in 600000 1000000 1500000 2000000 2500000 best; do
        ckpt="LAB5_${SID}_task3_${milestone}.pt"
        echo
        echo ">>> Snapshot: ${milestone}  ($(basename "$ckpt"))"
        $PY test_model_task3.py --model-path "$ckpt"
    done
}

case "${1:-all}" in
    task1) run_task1 ;;
    task2) run_task2 ;;
    task3) run_task3 ;;
    all)   run_task1; run_task2; run_task3 ;;
    *)     echo "Unknown target: $1 (use task1 | task2 | task3 | all)"; exit 1 ;;
esac
