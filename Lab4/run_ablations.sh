#!/bin/bash
# Launcher for Lab4 ablations on gpu9.
# Runs two experiments in parallel on CUDA 0 and CUDA 1, sequential per GPU.

set -u
cd /project3/chueating/DLP/Lab4
source /project3/chueating/env.sh
mkdir -p logs ckpts

COMMON="--DR dataset --batch_size 12 --num_epoch 70 --num_workers 4 --per_save 5 --lr 1e-3"

run_one () {
  local gpu=$1
  local tag=$2
  shift 2
  local extra="$*"
  local save="ckpts/${tag}"
  local log="logs/${tag}.log"
  mkdir -p "${save}"
  echo "[$(date +%H:%M:%S)] GPU${gpu} starting ${tag} :: ${extra}" | tee -a logs/launcher.log
  CUDA_VISIBLE_DEVICES=${gpu} .venv/bin/python src/Trainer.py \
      ${COMMON} --save_root "${save}" --run_name "${tag}" ${extra} \
      > "${log}" 2>&1
  echo "[$(date +%H:%M:%S)] GPU${gpu} finished ${tag}" | tee -a logs/launcher.log
}

# GPU 0 lane: A then C
(
  run_one 0 "A_cyclic_multistep"   --kl_anneal_type Cyclical  --lr_scheduler multistep
  run_one 0 "C_none_multistep"     --kl_anneal_type None      --lr_scheduler multistep
) &
PID0=$!

# GPU 1 lane: B then D
(
  run_one 1 "B_mono_multistep"     --kl_anneal_type Monotonic --lr_scheduler multistep
  run_one 1 "D_cyclic_cosine"      --kl_anneal_type Cyclical  --lr_scheduler cosine
) &
PID1=$!

wait $PID0 $PID1
echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a logs/launcher.log
