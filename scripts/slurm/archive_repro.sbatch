#!/bin/bash
# Persistent copies (scratch is purged after 28 days): for each repro run, copy the best-Val400 and final
# checkpoints (dereferenced) + run.json, epochs.csv, metrics*, preds*, logs into ~/fidip_artifacts/runs/<RUN_ID>/,
# then verify the checkpoint SHA256 against run.json. Also copies $R/models/SHA256SUMS + PROVENANCE.txt.
# Usage: sbatch scripts/slurm/archive_repro.sbatch R001 R002 R010 ...   (run-id prefixes)
#SBATCH --job-name=rp_archive
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
R=/scratch/purohitham.v/fidip; A=$HOME/fidip_artifacts
mkdir -p $A/runs $A/models
cp -p $R/models/SHA256SUMS $R/models/PROVENANCE.txt $A/models/
for p in "$@"; do
  for d in $R/runs/${p}_*; do
    [[ -d $d ]] || continue
    rid=$(basename $d); dst=$A/runs/$rid; mkdir -p $dst
    find $d -maxdepth 1 -type f \( -name 'run.json' -o -name 'epochs.csv' -o -name 'metrics_*' -o -name 'preds_*' \
      -o -name 'train.log' -o -name 'eval_*.log' \) -exec cp -p {} $dst/ \;
    for c in ckpt_best_val400.pth ckpt_final.pth; do
      [[ -e $d/$c ]] && cp -L $d/$c $dst/$c
    done
    python3 -I - "$d/run.json" "$dst" <<'EOF'
import hashlib, json, os, sys
rj, dst = json.load(open(sys.argv[1])), sys.argv[2]
for name, f in (('best_val400', 'ckpt_best_val400.pth'), ('final', 'ckpt_final.pth')):
    p = os.path.join(dst, f)
    if os.path.isfile(p):
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        ok = h == rj['checkpoints'][name]['sha256']
        print('%s %s sha256 %s %s' % (os.path.basename(dst), f, h[:16], 'OK' if ok else 'MISMATCH'))
        assert ok
EOF
  done
done
du -sh $A/runs
