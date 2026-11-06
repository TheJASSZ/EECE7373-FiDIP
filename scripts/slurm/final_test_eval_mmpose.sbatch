#!/bin/bash
# FINAL Test100 evaluation of pre-registered MMPose runs (GT boxes, flip test as configured).
#   * refuses unless the shared gate scripts/prereg.py registers the RUN_ID and the snapshot of
#     results/PREREGISTRATION.md has the Erratum section (checked here and again in eval_ckpt.py);
#   * evaluates exactly the checkpoint recorded in run.json (the Val400-selected epoch) and
#     verifies its SHA256 against run.json before running;
#   * writes preds_test100.json and metrics_test100.json (eval/eval_coco.py, paper + fixed).
# Required env: RUN_ID or RUN_IDS="a b c".  Each run is evaluated at most once (refuses if preds exist).
#SBATCH --job-name=im_final_test_mmpose
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
RUN_IDS=${RUN_IDS:-${RUN_ID:?set RUN_ID or RUN_IDS}}
R=/scratch/purohitham.v/fidip
# D19: run from a per-job code snapshot (a concurrent sync must never rewrite scripts under a running job)
SNAP=$R/snapshots/${SLURM_JOB_ID:-manual_$$}
mkdir -p $SNAP && rsync -a --exclude=.git $R/repo/ $SNAP/ || { echo "D19 snapshot failed"; exit 1; }
export REPO=$SNAP
PREREG=$REPO/results/PREREGISTRATION.md
grep -q "^### Erratum" $PREREG || { echo "REFUSED: $PREREG lacks the Erratum section"; exit 2; }
module load miniconda3/25.9.1 cuda/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $R/envs/mmpose
export PYTHONNOUSERSITE=1   # never let ~/.local site-packages shadow the env (orchestrator note)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}
export PYTHONPATH=$REPO/mmpose_exps:${PYTHONPATH:-}
python -c "import sys; sys.path.insert(0,'$REPO/scripts'); import prereg; print('allowed prefixes:', sorted(prereg.allowed_prefixes('$PREREG')))"
cd $R/mmpose
for RUN_ID in $RUN_IDS; do
  RUN_DIR=$R/runs/$RUN_ID
  python -c "import sys; sys.path.insert(0,'$REPO/scripts'); import prereg; sys.exit(0 if prereg.is_registered('$RUN_ID', prereg.allowed_prefixes('$PREREG')) else 2)" \
    || { echo "REFUSED: $RUN_ID is not registered for Test100"; continue; }
  test ! -e $RUN_DIR/preds_test100.json || { echo "REFUSED: $RUN_DIR/preds_test100.json exists (one shot only)"; continue; }
  read CKPT SHA < <(python -c "import json;d=json.load(open('$RUN_DIR/run.json'));print(d['checkpoint'],d['checkpoint_sha256'])")
  echo "run=$RUN_ID ckpt=$CKPT job=${SLURM_JOB_ID:-} node=$(hostname) start=$(date -Is)"
  echo "$SHA  $CKPT" | sha256sum -c - || { echo "REFUSED: checkpoint SHA256 mismatch for $RUN_ID"; continue; }
  python $REPO/mmpose_exps/tools/eval_ckpt.py $RUN_DIR/config.py $CKPT --split test100 \
    --out-dir $RUN_DIR --run-id $RUN_ID --prereg $PREREG
  python $REPO/eval/eval_coco.py --gt $R/data/splits/test100.json --pred $RUN_DIR/preds_test100.json \
    --out $RUN_DIR/metrics_test100.json --per-image $RUN_DIR/per_image_test100.json --run-id $RUN_ID
done
echo "done $(date -Is)"
