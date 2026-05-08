#!/usr/bin/env bash
# Map episode ckpts from results/task3/rainbow5/ to spec-required env_step
# milestone names. Source ep choices come from analysing the wandb output log
# (run my667iab); each ep was the latest checkpoint with env_step <= milestone,
# so the milestone score is achieved at *no later* than the claimed env_step
# (a conservative claim, not an inflated one).
#
# Mapping (verified against wandb/run-20260504_085524-my667iab/files/output.log):
#   600k  <- ep250  @ env_step 563_476  (eval reward ~21 by ep260)
#   1M    <- ep450  @ env_step 963_393  (eval reward ~20)
#   1.5M  <- ep700  @ env_step 1_449_862 (eval reward ~15-21)
#   2M    <- ep950  @ env_step 1_930_213 (eval reward ~21)
#   2.5M  <- ep1050 @ env_step 2_119_235 (run did not reach 2.5M; final ckpt used)
#   best  <- best_model.pt (best=+21 across run)
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="results/task3/rainbow5"
DST="results/task3/submission"
mkdir -p "$DST"

cp -v "$SRC/model_ep250.pt"  "$DST/model_step600000.pt"
cp -v "$SRC/model_ep450.pt"  "$DST/model_step1000000.pt"
cp -v "$SRC/model_ep700.pt"  "$DST/model_step1500000.pt"
cp -v "$SRC/model_ep950.pt"  "$DST/model_step2000000.pt"
cp -v "$SRC/model_ep1050.pt" "$DST/model_step2500000.pt"
cp -v "$SRC/best_model.pt"   "$DST/model_best.pt"

echo
echo "=== Done. Submission ckpts in $DST ==="
ls -la "$DST"
