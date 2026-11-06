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


def config_roles():
    """run_id -> role from results/configs.json (authoritative when present)."""
    p = os.path.join(REPO, "results", "configs.json")
    roles = {}
    if os.path.exists(p):
        for c in json.load(open(p)).get("configs", {}).values():
            for rid in c.get("run_ids", []):
                roles[rid] = c.get("role")
    return roles


def test_allowed(run_id, reg, pre, roles=None):
    """Reproduction targets (released checkpoints, official-recipe retrains) or pre-registered.
    Explicit roles in results/configs.json win over the run-name heuristics."""
    if run_id in pre:
        return True, "pre-registered"
    roles = config_roles() if roles is None else roles
    if run_id in roles:
        if roles[run_id] in ("released", "retrain"):
            return True, f"configs.json role '{roles[run_id]}' (reproduction target)"
        return False, f"configs.json role '{roles[run_id]}', not pre-registered -> withheld"
    row = reg.get(run_id, {})
    variant = (row.get("variant") or "").lower()
    if "released" in run_id or "released" in variant:
        return True, "released checkpoint (reproduction target)"
    if re.search(r"_retrain(_|$)", run_id) or variant == "retrain":
        return True, "official-recipe retrain (reproduction target)"
    return False, "not pre-registered -> withheld"


def derived_splits():
    """Sensitivity GTs derived from the official splits (written under $R/analysis/splits_derived):
    test100_noleak = Test100 minus images that are exact (md5) or near-exact (dHash<=10, pose OKS>=0.85)
    copies of annotated official training images (repro's $R/data/splits/VERIFY.json). The ids are read,
    never hard-coded. Annotation id 0 (test0) stays, so both protocols still apply."""
    vpath = os.path.join(R, "data", "splits", "VERIFY.json")
    out = {}
    if not os.path.exists(vpath):
        return out
    ov = json.load(open(vpath)).get("overlap", {})
    ids = set()
    for k, v in ov.items():
        if k.startswith("train_x_test100_by_md5_pairs") or k.startswith("test100_x_train_near_dups"):
            ids.update(int(x[0]) for x in v)
    if not ids:
        return out
    gt = json.load(open(os.path.join(R, "data", "splits", "test100.json")))
    sub = dict(gt, images=[im for im in gt["images"] if im["id"] not in ids],
               annotations=[x for x in gt["annotations"] if x["image_id"] not in ids])
    d = os.path.join(R, "analysis", "splits_derived")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "test100_noleak.json")
    if not os.path.exists(p) or json.load(open(p)) != sub:
        json.dump(sub, open(p, "w"))
    json.dump({"test100_noleak": {"excluded_test_ids": sorted(ids), "n_images": len(sub["images"]),
                                  "source": vpath}}, open(os.path.join(d, "README.json"), "w"), indent=1)
    out["test100_noleak"] = (p, "test100")
    return out


def eval_derived(run_id, pred, split, tag, od, derived, rows, force=False):
    """Evaluate `pred` on every derived GT of `split` (only reached after the Test100 gate passed)."""
    for dname, (dgt, parent) in derived.items():
        if parent != split or tag != "":
            continue
        dm, dp = os.path.join(od, f"metrics_{dname}.json"), os.path.join(od, f"per_image_{dname}.json")
        ps, gs = sha(pred), sha(dgt)
        if os.path.exists(dm) and not force:
            old = json.load(open(dm))
            if (old.get("pred", {}).get("sha256") == ps and old.get("gt", {}).get("sha256") == gs
                    and old.get("evaluator") == f"eval/eval_coco.py v{ev.EVALUATOR_VERSION}"):
                rows.append({"run_id": run_id, "split": dname, "pred": pred, "status": "up to date (derived)",
                             "AP": old["metrics"]["AP"], "AP_fixed": old["metrics_fixed"]["AP"],
                             "AP_80_95": old["metrics"]["AP_80_95"], "n_images": old["gt"]["n_images"]})
                continue
        ev.main(["--gt", dgt, "--pred", pred, "--out", dm, "--per-image", dp, "--run-id", run_id, "--split", dname,
                 "--quiet"])
        dd = json.load(open(dm))
        rows.append({"run_id": run_id, "split": dname, "pred": pred, "status": "evaluated (derived)",
                     "AP": dd["metrics"]["AP"], "AP_fixed": dd["metrics_fixed"]["AP"],
                     "AP_80_95": dd["metrics"]["AP_80_95"], "n_images": dd["gt"]["n_images"]})
        print(f"{run_id:40s} {dname:15s} AP {100 * dd['metrics']['AP']:6.2f}  fixed {100 * dd['metrics_fixed']['AP']:6.2f}")


def cmd_eval_all(a):
    reg, pre = registry(), prereg_ids()
    derived = derived_splits()
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
                eval_derived(run_id, pred, split, tag, od, derived, rows, a.force)
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
            print(f"{run_id:40s} {split + tag:15s} AP {100 * m['metrics']['AP']:6.2f}  fixed {100 * m['metrics_fixed']['AP']:6.2f}"
                  f"  AP80-95 {100 * m['metrics']['AP_80_95']:6.2f}")
            eval_derived(run_id, pred, split, tag, od, derived, rows, a.force)
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
