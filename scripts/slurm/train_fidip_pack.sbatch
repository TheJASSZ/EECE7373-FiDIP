#!/bin/bash
# Run several FiDIP trainings concurrently on ONE GPU (D14 packing), each followed by its evaluation.
# Each argument is one quoted run spec "RUN_ID SEED LAMBDA [EVAL_SETS] [extra yacs opts...]"
# (EVAL_SETS: quoted with commas, e.g. test100,val400 or val400 or none; default test100,val400).
# Example:
#   sbatch -p gpu --gres=gpu:h200:1 -J rp_R010-R012 scripts/slurm/train_fidip_pack.sbatch \
#     "R010_fidip-w48_lambda5e-4_s1 1 0.0005" "R011_fidip-w48_lambda5e-4_s2 2 0.0005" ...
# Per-process GPU memory is sampled every 30 s into $R/slurm_logs/<job>_gpumem.csv.
#SBATCH --job-name=rp_train_pack
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -uo pipefail
R=/scratch/purohitham.v/fidip
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
MEMLOG=$R/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpumem.csv
( echo "time,pid,used_memory_MiB"; while true; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | sed "s/^/$(date +%H:%M:%S),/"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | sed "s/^/$(date +%H:%M:%S),total,/"
    sleep 30; done ) > "$MEMLOG" 2>&1 &
MON=$!
cd $R/repo
pids=(); names=()
for spec in "$@"; do
  read -r RUN_ID SEED LAMBDA EVAL REST <<< "$spec"
  EVAL=${EVAL:-test100,val400}
  mkdir -p $R/runs/$RUN_ID
  echo "[pack] start $RUN_ID seed=$SEED lambda=$LAMBDA eval=$EVAL extra='${REST:-}'"
  bash scripts/train_fidip.sh "$RUN_ID" "$SEED" "$LAMBDA" "${EVAL//,/ }" ${REST:-} > $R/runs/$RUN_ID/job_${SLURM_JOB_ID}.log 2>&1 &
  pids+=($!); names+=("$RUN_ID")
  sleep 20   # stagger start-up (dataset loading, cudnn benchmark)
done
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[pack] ${names[$i]} OK"; else echo "[pack] ${names[$i]} FAILED"; fail=1; fi
  tail -4 $R/runs/${names[$i]}/job_${SLURM_JOB_ID}.log
done
kill $MON 2>/dev/null
echo "peak per-pid GPU MiB:"; awk -F, 'NR>1 && $2!="total"{if($3>m[$2])m[$2]=$3} END{for(p in m)print "  pid "p": "m[p]" MiB"}' "$MEMLOG"
echo "done at $(date)"
exit $fail
