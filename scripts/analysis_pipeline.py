#!/usr/bin/env python
"""analysis_pipeline.py -- cluster-side driver for the analysis stream (run inside sbatch).

  python scripts/analysis_pipeline.py eval-all [--force]
      Evaluate every $R/runs/<RUN_ID>/preds_<split>.json with eval/eval_coco.py, writing
      $R/analysis/eval/<RUN_ID>/{metrics,per_image}_<split>.json plus index.csv.
      - Skips files whose (pred sha256, GT sha256, evaluator version) are unchanged.
      - Test100 (and validate500, which contains it) is evaluated only for released/retrain
        reproduction targets and RUN_IDs listed in results/PREREGISTRATION.md (decision D1). Other
        such files are skipped and listed as 'withheld', so nobody can peek.
  python scripts/analysis_pipeline.py bootstrap --spec SPEC.json --gt test100 --out NAME
      Run bootstrap_ci.py contrasts on a declared spec, writing $R/analysis/bootstrap/bootstrap_<NAME>.json.

Paths: R defaults to /scratch/purohitham.v/fidip (override with $R).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
import eval_coco as ev  # noqa: E402

R = os.environ.get("R", "/scratch/purohitham.v/fidip")
RUN_RE = re.compile(r"\bR\d{3}[A-Za-z0-9_.\-]*")
TEST_LIKE = ("test100", "validate100", "validate500")
SPLIT_RE = re.compile(r"^preds_(test100|val400|validate500|validate100)(.*)\.json$")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prereg_ids():
    ids = set()
    p = os.path.join(REPO, "results", "PREREGISTRATION.md")
    if os.path.exists(p):
        for line in open(p):
            if line.strip().startswith("|"):
                ids.update(m.group(0).rstrip(".,;") for m in RUN_RE.finditer(line))
    return ids


def registry():
    p = os.path.join(REPO, "results", "registry.csv")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return {r["run_id"]: r for r in csv.DictReader(f) if r.get("run_id")}


def test_allowed(run_id, reg, pre):
    """Reproduction targets (released checkpoints, official-recipe retrains) or pre-registered."""
    if run_id in pre:
        return True, "pre-registered"
    row = reg.get(run_id, {})
    variant = (row.get("variant") or "").lower()
    if "released" in run_id or "released" in variant:
        return True, "released checkpoint (reproduction target)"
    if re.search(r"_retrain(_|$)", run_id) or variant == "retrain":
        return True, "official-recipe retrain (reproduction target)"
    return False, "not pre-registered -> withheld"


def cmd_eval_all(a):
    reg, pre = registry(), prereg_ids()
    out_root = os.path.join(R, "analysis", "eval")
    os.makedirs(out_root, exist_ok=True)
    rows = []
    for pred in sorted(glob.glob(os.path.join(R, "runs", "*", "preds_*.json"))):
        run_id = os.path.basename(os.path.dirname(pred))
        m = SPLIT_RE.match(os.path.basename(pred))
        if not m:
            rows.append({"run_id": run_id, "split": "?", "pred": pred, "status": "unrecognised file name"})
            continue
        split, tag = m.group(1), m.group(2)          # e.g. preds_val400_noflip.json -> val400, _noflip
        gt = os.path.join(R, "data", "splits", f"{split}.json")
        row = {"run_id": run_id, "split": split + tag, "pred": pred, "status": ""}
        if not os.path.exists(gt):
            row["status"] = f"no GT {gt}"
            rows.append(row)
            continue
        # validate500 contains Test100, so it is gated exactly like Test100 (no back door to Test100 numbers)
        if split in TEST_LIKE:
            ok, why = test_allowed(run_id, reg, pre)
            if not ok:
                row["status"] = why
                rows.append(row)
                print(f"WITHHELD {run_id}/{split}: {why}")
                continue
        od = os.path.join(out_root, run_id)
        os.makedirs(od, exist_ok=True)
        mpath = os.path.join(od, f"metrics_{split}{tag}.json")
        ppath = os.path.join(od, f"per_image_{split}{tag}.json")
        ps, gs = sha(pred), sha(gt)
        if os.path.exists(mpath) and os.path.exists(ppath) and not a.force:
            old = json.load(open(mpath))
            if (old.get("pred", {}).get("sha256") == ps and old.get("gt", {}).get("sha256") == gs
                    and old.get("evaluator") == f"eval/eval_coco.py v{ev.EVALUATOR_VERSION}"):
                row.update(status="up to date", AP=old["metrics"]["AP"], AP_fixed=old["metrics_fixed"]["AP"],
                           AP_80_95=old["metrics"]["AP_80_95"], n_images=old["gt"]["n_images"])
                rows.append(row)
                continue
        try:
            rc = ev.main(["--gt", gt, "--pred", pred, "--out", mpath, "--per-image", ppath, "--run-id", run_id,
                          "--split", split + tag, "--quiet"])
            m = json.load(open(mpath))
            row.update(status="evaluated" if rc == 0 else f"rc={rc}", AP=m["metrics"]["AP"],
                       AP_fixed=m["metrics_fixed"]["AP"], AP_80_95=m["metrics"]["AP_80_95"],
                       n_images=m["gt"]["n_images"], n_pred=m["pred"]["n_in"],
                       n_dropped=m["pred"]["n_dropped_not_in_gt"],
                       n_missing=m["pred"]["n_gt_images_without_pred"], warnings=" | ".join(m["warnings"]))
            print(f"{run_id:40s} {split:12s} AP {100 * m['metrics']['AP']:6.2f}  fixed {100 * m['metrics_fixed']['AP']:6.2f}"
                  f"  AP80-95 {100 * m['metrics']['AP_80_95']:6.2f}")
        except Exception as e:  # keep going; record the failure
            row["status"] = f"ERROR {type(e).__name__}: {e}"
            print(f"ERROR {run_id}/{split}: {e}")
        rows.append(row)
    keys = ["run_id", "split", "status", "AP", "AP_fixed", "AP_80_95", "n_images", "n_pred", "n_dropped", "n_missing",
            "warnings", "pred"]
    with open(os.path.join(out_root, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[eval-all] {len(rows)} prediction files at {dt.datetime.now().isoformat(timespec='seconds')} "
          f"-> {out_root}/index.csv")


def cmd_bootstrap(a):
    import bootstrap_ci as bc
    gt = os.path.join(R, "data", "splits", f"{a.gt}.json") if not a.gt.endswith(".json") else a.gt
    out = os.path.join(R, "analysis", "bootstrap", f"bootstrap_{a.out}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return bc.main(["contrasts", "--gt", gt, "--spec", a.spec, "--out", out] + (["--B", str(a.B)] if a.B else []))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("eval-all")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("bootstrap")
    s.add_argument("--spec", required=True)
    s.add_argument("--gt", default="test100")
    s.add_argument("--out", required=True)
    s.add_argument("--B", type=int, default=None)
    a = ap.parse_args(argv)
    return {"eval-all": cmd_eval_all, "bootstrap": cmd_bootstrap}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
