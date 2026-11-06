#!/usr/bin/env bash
# Run a declared contrast spec on Test100 and Test100-clean (both protocols each; Holm per protocol).
# Contrasts whose prediction files are missing are skipped and listed; rerun once they exist, so Holm
# covers the full declared family.
#   sbatch scripts/slurm/analysis.sbatch script scripts/analysis/run_contrasts.sh results/contrasts/<spec>.json
set -euo pipefail
export R=${R:-/scratch/purohitham.v/fidip}
SPEC=$1; NAME=$(basename "$SPEC" .json); RUNSPEC=$R/analysis/bootstrap/${NAME}.runnable.json
mkdir -p $R/analysis/bootstrap
python - "$SPEC" "$RUNSPEC" <<'PY'
import json, os, sys
spec = json.load(open(sys.argv[1]))
keep, skip = [], []
for c in spec["contrasts"]:
    files = [os.path.expandvars(p) for k in ("a", "b") for p in (c[k] if isinstance(c[k], list) else [c[k]])]
    (keep if all(os.path.exists(f) for f in files) else skip).append(c)
spec["contrasts"] = keep
spec["skipped_missing_inputs"] = [c["name"] for c in skip]
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print("run:", [c["name"] for c in keep], "| skipped (missing preds):", [c["name"] for c in skip])
sys.exit(0 if keep else 3)
PY
python scripts/analysis_pipeline.py bootstrap --spec "$RUNSPEC" --gt test100 --out "${NAME}_test100"
python scripts/analysis_pipeline.py bootstrap --spec "$RUNSPEC" --gt test100_clean --out "${NAME}_test100_clean"
