#!/usr/bin/env bash
# IMPROVE stream: submit one training run of scripts/slurm/train_mmpose.sbatch from the Mac.
# usage: bash scripts/launch_mmpose.sh RUN_ID CONFIG SEED "CFG_OPTS" [extra sbatch args...]
#   e.g. bash scripts/launch_mmpose.sh R102_mmp-vitpose-l256_base_s1 \
#          mmpose_exps/configs/syrip_vitpose-l_udp-256x192.py 1 "" -p gpu --gres=gpu:a100:1 -t 01:30:00
# Parameters travel in the submitting environment (--export=ALL), so cfg-options may contain commas.
# Prints the Slurm job id.
set -euo pipefail
RUN_ID=$1; CONFIG=$2; SEED=$3; CFG_OPTS=$4; shift 4
ssh -o BatchMode=yes -o ConnectTimeout=15 explorer \
  "cd /scratch/purohitham.v/fidip/repo && test -f $CONFIG && RUN_ID='$RUN_ID' CONFIG='$CONFIG' SEED='$SEED' CFG_OPTS='$CFG_OPTS' \
   sbatch --parsable -J 'im_$RUN_ID' --export=ALL $* scripts/slurm/train_mmpose.sbatch"
