#!/usr/bin/env bash
# Task 3 Enhanced DQN on ALE/Pong-v5: 5/6 of Rainbow (no C51).
#   DDQN + PER + n-step=3 + Dueling + NoisyNet
#
# Hyperparameters mirror jayin92's final run (`results_pong_full_3` / v22):
# best ckpt at env_step=200k, hitting reward 20 well within the ≤600k bonus.
#
# Diff vs run_pong_v2.sh (Task 2 vanilla baseline):
#   --replay-buffer-type uniform -> prioritized            (+ PER hyperparams)
#   --loss-fn-type mse           -> huber                  (PER stability)
#   --double-dqn                 (off -> on)
#   --n-step                     1 -> 3                    (Rainbow default)
#   --dueling-dqn                (off -> on)               (Wang 2016)
#   --noisy-net                  (off -> on)               (Fortunato 2018)
#   --train-per-step             1 -> 4                    (replay_ratio 0.25 -> 1.0)
#   --memory-size                100000 -> 50000           (more on-policy with high RR)
#   --target-update-frequency    1000 -> 500               (faster sync at 4x update rate)
#   --replay-start-size          10000 -> 5000             (start learning earlier)
#
# Sample-efficiency math: train_frequency=4 + train_per_step=4 means 4 gradient
# updates per env_step (replay_ratio = 1.0). Bumping RR 4x is the dominant lever
# for the ≤600k env_step convergence target; the smaller buffer (50k) keeps
# experience fresher to match the higher reuse rate.
#
# NoisyNet: --epsilon-* flags are ignored when --noisy-net is on; the agent
# zeroes epsilon internally so wandb's rollout/epsilon log reads 0 (exploration
# is purely network noise).
#
# PER hyperparameters (Schaul et al. 2016 / Rainbow defaults):
#   alpha=0.6, beta=0.4, beta_increment=1e-6
#   At RR=1.0 over 600k env_steps -> 600k updates; beta reaches 1.0 at ~600k
#   updates, exactly when we expect convergence. Earlier IS-weight correction
#   is partial-by-design (Schaul Fig.3).
set -euo pipefail
cd "$(dirname "$0")/.."

python dqn.py \
    --env-name ALE/Pong-v5 \
    --wandb-project dlp-lab5-task2 \
    --wandb-run-name pong-task3-rainbow5 \
    --save-dir ./results/task3/rainbow5/ \
    --episodes 10000 \
    --max-episode-steps 100000 \
    --batch-size 32 \
    --lr 1e-4 \
    --memory-size 50000 \
    --replay-start-size 5000 \
    --replay-buffer-type prioritized \
    --per-alpha 0.6 \
    --per-beta 0.4 \
    --per-beta-increment 1e-6 \
    --double-dqn \
    --n-step 3 \
    --dueling-dqn \
    --noisy-net \
    --discount-factor 0.99 \
    --target-update-frequency 500 \
    --loss-fn-type huber \
    --huber-beta 1.0 \
    --train-per-step 4 \
    --train-frequency 4 \
    --frame-stack 4 \
    --eval-interval 20 \
    --checkpoint-interval 50 \
    "$@"
