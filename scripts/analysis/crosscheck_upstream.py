#!/usr/bin/env python
"""Compare eval_coco.py (paper protocol, rescored) with upstream's own printed COCOeval summary
($R/runs/<RUN_ID>/metrics_upstream_<split>.txt) for every run/split that has both. Upstream prints
3 decimals, so a match means |ours - printed| <= 0.0005 on all 10 stats. Writes
$R/analysis/tables/crosscheck_upstream.csv."""
import csv, glob, json, os, re, sys
R = os.environ.get("R", "/scratch/purohitham.v/fidip")
NAMES = ["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"]
rows, ok_all = [], True
for txt in sorted(glob.glob(f"{R}/runs/*/metrics_upstream_*.txt")):
    rid = os.path.basename(os.path.dirname(txt))
    split = os.path.basename(txt)[len("metrics_upstream_"):-4]
    vals = [float(x) for x in re.findall(r"\] = (-?\d+\.\d+)", open(txt).read())][:10]
    mj = f"{R}/analysis/eval/{rid}/metrics_{split}.json"
    if len(vals) != 10 or not os.path.exists(mj):
        continue
    m = json.load(open(mj))["metrics"]
    ours = [-1.0 if m[k] is None else m[k] for k in NAMES]
    d = max(abs(a - b) for a, b in zip(ours, vals))
    ok = d <= 0.0005 + 1e-12
    ok_all &= ok
    rows.append({"run_id": rid, "split": split, "max_abs_diff": round(d, 6), "match": ok,
                 **{f"ours_{k}": round(v, 5) for k, v in zip(NAMES, ours)},
                 **{f"upstream_{k}": v for k, v in zip(NAMES, vals)}})
    print(f"{rid:32s} {split:16s} AP ours {ours[0]:.5f} upstream {vals[0]:.3f}  max|d| {d:.5f}  {'MATCH' if ok else 'MISMATCH'}")
os.makedirs(f"{R}/analysis/tables", exist_ok=True)
with open(f"{R}/analysis/tables/crosscheck_upstream.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id"]); w.writeheader(); w.writerows(rows)
print("ALL MATCH" if ok_all and rows else "CHECK FAILED", len(rows), "pairs")
sys.exit(0 if ok_all else 1)
