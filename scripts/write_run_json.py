#!/usr/bin/env python3
"""Write/merge $R/runs/<RUN_ID>/run.json with provenance (stdlib only; called from sbatch scripts).

Records: run id, Slurm job id/name/partition, node, GPU (nvidia-smi), git state pushed to the cluster
(.repo_version.json: HEAD, HEAD^{tree}, dirty flags), a sha256 over the upstream code actually run
(lib/, tools/, experiments/), checkpoint sha256s, the command lines, and free-form key=value fields.
Repeated calls merge into the same file ("commands" and "checkpoints" accumulate).

  python3 -I scripts/write_run_json.py --run-dir DIR --run-id ID [--cmd "..."] [--ckpt name=path]
         [--set key=value] [--stamp start|end]
"""
import argparse
import hashlib
import json
import os
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = os.path.join(REPO, 'third_party', 'Infant-Pose-Estimation')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def code_sha256():
    """sha256 over sorted (relative path, file sha256) of upstream lib/tools/experiments (*.py, *.yaml)."""
    h = hashlib.sha256()
    for sub in ('lib', 'tools', 'experiments'):
        for root, _, files in sorted(os.walk(os.path.join(UP, sub))):
            for f in sorted(files):
                if f.endswith(('.py', '.yaml')):
                    p = os.path.join(root, f)
                    h.update(('%s %s\n' % (os.path.relpath(p, UP), sha256_file(p))).encode())
    return h.hexdigest()


def gpu():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out.splitlines()
    except Exception as e:  # noqa: BLE001
        return ['n/a (%s)' % e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--cmd', action='append', default=[])
    ap.add_argument('--ckpt', action='append', default=[], help='name=path')
    ap.add_argument('--set', action='append', default=[], help='key=value (value parsed as JSON if possible)')
    ap.add_argument('--stamp', choices=['start', 'end'])
    a = ap.parse_args()
    os.makedirs(a.run_dir, exist_ok=True)
    path = os.path.join(a.run_dir, 'run.json')
    rj = json.load(open(path)) if os.path.isfile(path) else {}
    rj['run_id'] = a.run_id
    rj.setdefault('slurm', {}).update({k: os.environ.get(v) for k, v in (
        ('job_id', 'SLURM_JOB_ID'), ('job_name', 'SLURM_JOB_NAME'), ('partition', 'SLURM_JOB_PARTITION'),
        ('node', 'SLURMD_NODENAME'), ('gpus', 'CUDA_VISIBLE_DEVICES'))})
    rj['gpu'] = gpu()
    vf = os.path.join(REPO, '.repo_version.json')
    rj['git_pushed'] = json.load(open(vf)) if os.path.isfile(vf) else 'unknown (.repo_version.json missing)'
    rj['upstream_code_sha256'] = code_sha256()
    rj.setdefault('commands', [])
    for c in a.cmd:
        if c not in rj['commands']:
            rj['commands'].append(c)
    rj.setdefault('checkpoints', {})
    for kv in a.ckpt:
        k, p = kv.split('=', 1)
        rj['checkpoints'][k] = {'path': p, 'sha256': sha256_file(p) if os.path.isfile(p) else None}
    for kv in a.set:
        k, v = kv.split('=', 1)
        try:
            rj[k] = json.loads(v)
        except ValueError:
            rj[k] = v
    if a.stamp:
        rj['time_' + a.stamp] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    json.dump(rj, open(path, 'w'), indent=1, sort_keys=True)


if __name__ == '__main__':
    main()
