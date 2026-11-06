#!/usr/bin/env bash
# Run a declared contrast spec on Test100 and Test100-clean (both protocols each; Holm per protocol).
#   sbatch scripts/slurm/analysis.sbatch script scripts/analysis/run_contrasts.sh results/contrasts/<spec>.json
set -euo pipefail
export R=${R:-/scratch/purohitham.v/fidip}
SPEC=$1; NAME=$(basename "$SPEC" .json)
for f in $(python - "$SPEC" <<'PY'
import json, os, sys
for c in json.load(open(sys.argv[1]))["contrasts"]:
    for p in (c["a"] if isinstance(c["a"], list) else [c["a"]]) + (c["b"] if isinstance(c["b"], list) else [c["b"]]):
        print(os.path.expandvars(p))
PY
); do test -f "$f" || { echo "MISSING $f -- contrast not run"; exit 3; }; done
python scripts/analysis_pipeline.py bootstrap --spec "$SPEC" --gt test100 --out "${NAME}_test100"
python scripts/analysis_pipeline.py bootstrap --spec "$SPEC" --gt test100_clean --out "${NAME}_test100_clean"
