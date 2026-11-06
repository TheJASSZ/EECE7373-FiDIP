#!/bin/bash
# IMPROVE stream: SEVERAL fine-tuning runs packed concurrently on ONE GPU (D14 budget policy).
# Each run is an independent process (scripts/run_one_mmpose.sh) with its own seed, run dir and log
# ($R/runs/<RUN_ID>/job_stdout.log). Per-process GPU memory and GPU utilisation are sampled every
# 60 s into $R/logs/pack/<jobid>_gpu.csv; each run dir gets pack_info.json naming its co-runners.
# Required env: RUNS="RUN_ID|CONFIG|SEED|CFG_OPTS;RUN_ID|CONFIG|SEED|CFG_OPTS;..." (CFG_OPTS may be
# empty and may contain spaces, but no ';' or '|').
# Memory per process (batch 32, measured): HRNet-W48 384x288 ~14.5 GB, ViT-L 256x192 ~12 GB,
# ViT-L 384x288 ~24 GB, ViT-H 256x192 ~21 GB, ViT-H 384x288 ~45 GB. Keep the sum below the card.
#SBATCH --job-name=im_pack_mmpose
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -uo pipefail
: "${RUNS:?set RUNS}"
R=/scratch/purohitham.v/fidip
REPO=$R/repo
module load miniconda3/25.9.1 cuda/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $R/envs/mmpose
export PYTHONNOUSERSITE=1   # never let ~/.local site-packages shadow the env (orchestrator note)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}
export PYTHONPATH=$REPO/mmpose_exps:${PYTHONPATH:-}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mkdir -p $R/logs/pack
MON=$R/logs/pack/${SLURM_JOB_ID}_gpu.csv
( echo "time,pid,used_memory_MiB,gpu_util_pct,gpu_mem_used_MiB"
  while true; do
    T=$(date +%H:%M:%S)
    U=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d ' ' | sed "s/^/$T,/;s/\$/,$U/"
    sleep 60
  done ) > $MON 2>/dev/null &
MONPID=$!
IFS=';' read -ra SPECS <<< "$RUNS"
NAMES=()
for spec in "${SPECS[@]}"; do IFS='|' read -r rid _ <<< "$spec"; NAMES+=("$rid"); done
PIDS=()
for spec in "${SPECS[@]}"; do
  IFS='|' read -r RUN_ID CONFIG SEED CFG_OPTS <<< "$spec"
  mkdir -p $R/runs/$RUN_ID
  printf '{"pack_job": "%s", "node": "%s", "co_runs": "%s", "gpu_monitor": "%s"}\n' \
    "${SLURM_JOB_ID:-}" "$(hostname)" "${NAMES[*]}" "$MON" > $R/runs/$RUN_ID/pack_info.json
  bash $REPO/scripts/run_one_mmpose.sh "$RUN_ID" "$CONFIG" "${SEED:-1}" "${CFG_OPTS:-}" \
    > $R/runs/$RUN_ID/job_stdout.log 2>&1 &
  PIDS+=($!)
  echo "started $RUN_ID pid=$! ($(date -Is))"
  sleep 20   # stagger start-up (checkpoint loading, dataset init)
done
RC=0
for i in "${!PIDS[@]}"; do
  wait ${PIDS[$i]}; rc=$?
  echo "finished ${NAMES[$i]} rc=$rc ($(date -Is))"; tail -2 $R/runs/${NAMES[$i]}/job_stdout.log
  [ $rc -ne 0 ] && RC=$rc
done
kill $MONPID 2>/dev/null
echo "gpu monitor: $MON"; awk -F, 'NR>1{m[$2]=($3>m[$2]?$3:m[$2])} END{for(p in m) print "  peak MiB pid " p ": " m[p]}' $MON
exit $RC
