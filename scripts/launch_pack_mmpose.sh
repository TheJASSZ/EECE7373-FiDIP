#!/usr/bin/env bash
# IMPROVE stream: submit several runs packed on one GPU (scripts/slurm/train_pack_mmpose.sbatch).
# usage: bash scripts/launch_pack_mmpose.sh PACK_NAME "RID|CONFIG|SEED|CFG_OPTS;RID|..." [sbatch args...]
#   e.g. bash scripts/launch_pack_mmpose.sh screenA "R101_x|mmpose_exps/configs/a.py|1|;R102_y|...|1|" \
#          -p gpu --gres=gpu:a100:1 -t 02:00:00
set -euo pipefail
NAME=$1; RUNS=$2; shift 2
ssh -o BatchMode=yes -o ConnectTimeout=15 explorer \
  "cd /scratch/purohitham.v/fidip/repo && RUNS='$RUNS' sbatch --parsable -J 'im_pack_$NAME' --export=ALL $* scripts/slurm/train_pack_mmpose.sbatch"
