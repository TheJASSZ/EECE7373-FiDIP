#!/bin/bash
# IMPROVE stream: R141 = the pre-registered 5-seed ensemble of arm B (imp-3): per-instance mean of the
# flip-tested keypoint coordinates of R136-R140 (eval/ensemble_preds.py, --weighting mean), scored with
# eval/eval_coco.py. Built for Test100 (gated + one shot) and, as supporting evidence, for Val400.
# CPU only. Optional env: ENS_ID, MEMBERS (space-separated RUN_IDs), SPLITS (default "val400 test100").
#SBATCH --job-name=im_ensemble
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
R=/scratch/purohitham.v/fidip
# D19: run from a per-job code snapshot
SNAP=$R/snapshots/${SLURM_JOB_ID:-manual_$$}
mkdir -p $SNAP && rsync -a --exclude=.git $R/repo/ $SNAP/ || { echo "D19 snapshot failed"; exit 1; }
export REPO=$SNAP
PREREG=$REPO/results/PREREGISTRATION.md
module load miniconda3/25.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $R/envs/mmpose
export PYTHONNOUSERSITE=1
ENS_ID=${ENS_ID:-R141_mmp-vitpose-h256_ens5}
MEMBERS=${MEMBERS:-"R136_mmp-vitpose-h256_conf_s2 R137_mmp-vitpose-h256_conf_s3 R138_mmp-vitpose-h256_conf_s4 R139_mmp-vitpose-h256_conf_s5 R140_mmp-vitpose-h256_conf_s6"}
OUT=$R/runs/$ENS_ID
mkdir -p $OUT
for SPLIT in ${SPLITS:-val400 test100}; do
  if [ "$SPLIT" = test100 ]; then
    grep -q "^### Erratum" $PREREG || { echo "REFUSED: $PREREG lacks the Erratum section"; exit 2; }
    python -c "import sys; sys.path.insert(0,'$REPO/scripts'); import prereg; p=prereg.allowed_prefixes('$PREREG'); print('allowed prefixes:', sorted(p)); sys.exit(0 if prereg.is_registered('$ENS_ID', p) else 2)" \
      || { echo "REFUSED: $ENS_ID not registered for Test100"; exit 2; }
    test ! -e $OUT/preds_test100.json || { echo "REFUSED: $OUT/preds_test100.json exists (one shot only)"; exit 3; }
  fi
  P=""
  for m in $MEMBERS; do f=$R/runs/$m/preds_$SPLIT.json; test -f $f || { echo "missing $f"; exit 4; }; P="$P $f"; done
  python $REPO/eval/ensemble_preds.py --gt $R/data/splits/$SPLIT.json --preds $P --out $OUT/preds_$SPLIT.json --weighting mean
  python $REPO/eval/eval_coco.py --gt $R/data/splits/$SPLIT.json --pred $OUT/preds_$SPLIT.json \
    --out $OUT/metrics_$SPLIT.json --per-image $OUT/per_image_$SPLIT.json --run-id $ENS_ID
done
python - "$OUT" "$ENS_ID" $MEMBERS <<'PY'
import hashlib, json, os, sys, time
out, ens, members = sys.argv[1], sys.argv[2], sys.argv[3:]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
R = '/scratch/purohitham.v/fidip/runs/'
mem = []
for m in members:
    rj = json.load(open(R + m + '/run.json'))
    mem.append(dict(run_id=m, checkpoint=rj['checkpoint'], checkpoint_sha256=rj['checkpoint_sha256'],
                    preds={s: sha(R + m + f'/preds_{s}.json') for s in ('val400', 'test100')
                           if os.path.isfile(R + m + f'/preds_{s}.json')}))
rj = dict(run_id=ens, stream='improve', kind='ensemble (pre-registered imp-3)',
          method='eval/ensemble_preds.py --weighting mean: per-instance mean of flip-tested keypoint '
                 'coordinates over the members; scores re-derived by eval_coco rescoring',
          members=mem, slurm_job_id=os.environ.get('SLURM_JOB_ID'), code_snapshot=os.environ.get('REPO'),
          metrics={s: json.load(open(f'{out}/metrics_{s}.json'))['metrics'] for s in ('val400', 'test100')
                   if os.path.isfile(f'{out}/metrics_{s}.json')},
          finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
json.dump(rj, open(out + '/run.json', 'w'), indent=1)
print(json.dumps({k: rj[k] for k in ('run_id', 'metrics')}, indent=1)[:1500])
PY
echo "done $(date -Is)"
