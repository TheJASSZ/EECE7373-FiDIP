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
import glob, json, os, sys
spec = json.load(open(sys.argv[1]))
keep, skip = [], []
def expand(p):
    p = os.path.expandvars(p)
    if any(ch in p for ch in "*?["):
        m = sorted(glob.glob(p))
        return m[0] if len(m) == 1 else None          # a pattern must name exactly one file
    return p if os.path.exists(p) else None
for c in spec["contrasts"]:
    ok = True
    for k in ("a", "b"):
        lst = c[k] if isinstance(c[k], list) else [c[k]]
        ex = [expand(p) for p in lst]
        ok &= all(ex)
        c[k] = ex if all(ex) else lst
    (keep if ok else skip).append(c)
spec["contrasts"] = keep
spec["skipped_missing_inputs"] = [c["name"] for c in skip]
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print("run:", [c["name"] for c in keep], "| skipped (missing preds):", [c["name"] for c in skip])
sys.exit(0 if keep else 3)
PY
python scripts/analysis_pipeline.py bootstrap --spec "$RUNSPEC" --gt test100 --out "${NAME}_test100"
python scripts/analysis_pipeline.py bootstrap --spec "$RUNSPEC" --gt test100_clean --out "${NAME}_test100_clean"
