#!/usr/bin/env python
"""Collect provenance for one MMPose run dir into run.json (+ val_curve.json, registry_row.csv).

run.json: run id, Slurm job/partition/node, GPU, seed, config (+ SHA256), git head/tree/dirty
(from $R/repo/GIT_INFO_mmpose.json, written by scripts/push_mmpose.sh at sync time), init
checkpoint (+ SHA256), selected checkpoint (best Val400 AP epoch) path + SHA256, per-epoch Val400
curve (MMPose CocoMetric; Val400 has no id-0 GT so it equals eval_coco.py there), and the
eval_coco.py headline Val400 metrics if metrics_val400.json exists.

Usage: python finalize_run.py RUN_DIR --run-id RID --config CFG --seed S [--started ISO]
"""
import argparse
import csv
import glob
import hashlib
import io
import json
import os
import os.path as osp
import re
import shutil
import socket
import subprocess
import time

GIT_INFO = '/scratch/purohitham.v/fidip/repo/GIT_INFO_mmpose.json'


def sha256(path, bs=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(bs), b''):
            h.update(chunk)
    return h.hexdigest()


def gpu_name():
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    except Exception:
        return None


def val_curve(run_dir):
    """Per-epoch Val400 metrics from every timestamped vis_data/scalars.json (resumes included)."""
    rows = {}
    for p in sorted(glob.glob(osp.join(run_dir, '2*', 'vis_data', 'scalars.json'))):
        with open(p) as f:
            for ln in f:
                d = json.loads(ln)
                if 'coco/AP' in d:
                    rows[int(d['step'])] = {k: d[k] for k in d if k.startswith('coco/')}
    return [dict(epoch=e, **rows[e]) for e in sorted(rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--config', required=True, help='frozen copy of the source config (RUN_DIR/config_src.py)')
    ap.add_argument('--config-name', required=True, help='basename of the repo config (mmpose_exps/configs/<name>)')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--started', default=None)
    ap.add_argument('--status', default='done')
    ap.add_argument('--notes', default='')
    args = ap.parse_args()
    rd = args.run_dir

    # canonical copies of the resolved config (mmengine dump) and the train log(s)
    dumped = osp.join(rd, args.config_name)
    if osp.isfile(dumped) and not osp.isfile(osp.join(rd, 'config.py')):
        shutil.copyfile(dumped, osp.join(rd, 'config.py'))
    logs = sorted(glob.glob(osp.join(rd, '2*', '2*.log')))
    if logs:
        with open(osp.join(rd, 'train.log'), 'w') as out:
            for lp in logs:
                with open(lp) as f:
                    out.write(f'# ==== {osp.relpath(lp, rd)} ====\n' + f.read())

    best = sorted(glob.glob(osp.join(rd, 'best_coco_AP_epoch_*.pth')))
    best_ckpt = best[-1] if best else None
    best_epoch = int(re.search(r'epoch_(\d+)\.pth$', best_ckpt).group(1)) if best_ckpt else None

    from mmengine.config import Config
    cfg = Config.fromfile(osp.join(rd, 'config.py'))
    curve = val_curve(rd)
    with open(osp.join(rd, 'val_curve.json'), 'w') as f:
        json.dump(curve, f, indent=1)

    git = json.load(open(GIT_INFO)) if osp.isfile(GIT_INFO) else {}
    m = {}
    mp = osp.join(rd, 'metrics_val400.json')
    if osp.isfile(mp):
        mv = json.load(open(mp))
        m = {k: mv['metrics'].get(k) for k in ('AP', 'AP50', 'AP75', 'AR', 'AP_80_95')}
    rj = dict(
        run_id=args.run_id, stream='improve', status=args.status,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        partition=os.environ.get('SLURM_JOB_PARTITION'), node=socket.gethostname(),
        gpu=gpu_name(), seed=args.seed,
        config='mmpose_exps/configs/' + args.config_name,
        config_src=osp.abspath(args.config), config_src_sha256=sha256(args.config),
        config_resolved=osp.join(rd, 'config.py'),
        config_resolved_sha256=sha256(osp.join(rd, 'config.py')),
        git_head=git.get('git_head'), git_tree=git.get('git_tree'), git_dirty=git.get('dirty'),
        git_dirty_files=git.get('dirty_files'), git_synced_at=git.get('synced_at'),
        init_checkpoint=cfg.get('load_from'),
        init_checkpoint_sha256=sha256(cfg.load_from) if cfg.get('load_from') else None,
        max_epochs=cfg.train_cfg.get('max_epochs'), batch_size=cfg.train_dataloader.batch_size,
        lr=cfg.optim_wrapper.optimizer.get('lr'), optimizer=cfg.optim_wrapper.optimizer.type,
        selection_rule='best Val400 coco/AP epoch (MMPose CocoMetric, checked every epoch)',
        best_epoch=best_epoch, checkpoint=best_ckpt,
        checkpoint_sha256=sha256(best_ckpt) if best_ckpt else None,
        val400_mmpose_best=max(curve, key=lambda r: r['coco/AP']) if curve else None,
        val400_eval_coco=m, started=args.started,
        finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    with open(osp.join(rd, 'run.json'), 'w') as f:
        json.dump(rj, f, indent=1)

    mm = re.match(r'R\d+_([^_]+)_(.+)_s(\d+)$', args.run_id)
    model, variant = (mm.group(1), mm.group(2)) if mm else ('', '')
    row = [args.run_id, rj['slurm_job_id'], 'improve', model, variant, args.seed,
           'mmpose_exps/configs/' + args.config_name,
           f"{git.get('git_head', '')[:12]}/tree:{git.get('git_tree', '')[:12]}"
           + ('+dirty' if git.get('dirty') else ''),
           rj['partition'], rj['gpu'], args.status,
           '' if m.get('AP') is None else f"{100 * m['AP']:.2f}", '',
           best_ckpt or '', rj['checkpoint_sha256'] or '', args.notes]
    buf = io.StringIO()
    csv.writer(buf).writerow(row)
    with open(osp.join(rd, 'registry_row.csv'), 'w') as f:
        f.write(buf.getvalue())
    print(json.dumps(rj, indent=1))


if __name__ == '__main__':
    main()
