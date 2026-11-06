#!/usr/bin/env python3
"""Append registry rows for runs that exist in the local mirror (results/cluster_runs/<RUN_ID>/run.json)
but have no row in results/registry.csv. Every value is read from run.json and from eval_coco metrics
(results/analysis_eval or the run mirror). Nothing is typed by hand. Idempotent: an existing run_id is
never duplicated."""
import csv, glob, json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(REPO, "results", "registry.csv")
MIRROR = os.path.join(REPO, "results", "cluster_runs")
TARGETS = ["R003", "R021", "R022", "R023", "R122"] + [f"R{i}" for i in range(131, 142)]

def first(d, *keys):
    for k in keys:
        cur = d
        for part in k.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur not in (None, "", []):
            return cur
    return ""

def metric(run_dir_name, split):
    for root in (os.path.join(REPO, "results", "analysis_eval"), MIRROR):
        for tag in ("", "_best"):
            f = os.path.join(root, run_dir_name, f"metrics_{split}{tag}.json")
            if os.path.exists(f):
                try:
                    return f"{100 * json.load(open(f))['metrics']['AP']:.2f}"
                except Exception:
                    pass
    return ""

rows = list(csv.DictReader(open(REG)))
fields = list(rows[0].keys()) if rows else ["run_id", "job_id", "stream", "model", "variant", "seed", "config", "git_sha",
                                            "partition", "gpu", "status", "val400_AP", "test100_AP", "ckpt_path", "ckpt_sha256", "notes"]
have = {r["run_id"] for r in rows}
new = []
for rj in sorted(glob.glob(os.path.join(MIRROR, "R*", "run.json"))):
    name = os.path.basename(os.path.dirname(rj))
    if name in have or not any(name.startswith(t) for t in TARGETS):
        continue
    d = json.load(open(rj))
    mm = name.split("_")[1] if "_" in name else ""
    new.append({
        "run_id": name,
        "job_id": first(d, "slurm_job_id", "job_id", "job.id", "slurm.job_id"),
        "stream": "improve" if "mmp" in name else "repro",
        "model": first(d, "model", "arch") or mm,
        "variant": "_".join(name.split("_")[2:-1]) if name.count("_") >= 2 else "",
        "seed": first(d, "seed"),
        "config": first(d, "config", "cfg"),
        "git_sha": first(d, "git_head", "git.head", "git_sha", "repo_version.git_head"),
        "partition": first(d, "partition", "slurm.partition"),
        "gpu": first(d, "gpu", "gpu_name", "train_gpu", "node_gpu"),
        "status": first(d, "status") or "completed",
        "val400_AP": metric(name, "val400"),
        "test100_AP": metric(name, "test100"),
        "ckpt_path": first(d, "checkpoint", "checkpoint.path", "ckpt", "ckpt_best_val400"),
        "ckpt_sha256": first(d, "checkpoint_sha256", "checkpoint.sha256", "ckpt_sha256", "ckpt_best_val400_sha256"),
        "notes": "backfilled from run.json by scripts/registry_backfill.py",
    })
for r in new:
    for k, v in list(r.items()):
        if isinstance(v, (dict, list)):
            r[k] = json.dumps(v)[:120]
with open(REG, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    for r in new:
        w.writerow(r)
print(f"appended {len(new)} rows:", ", ".join(r["run_id"] for r in new))
