#!/usr/bin/env python
"""Collect IMPROVE-stream run results into one table (eval/eval_coco.py numbers only).

Reads <root>/<RUN_ID>/{run.json, metrics_val400*.json} (e.g. results/cluster_runs after
scripts/sync_results.sh) and writes CSV + Markdown with AP, AP50, AP75, AR, AP@OKS .80-.95,
selected epoch, job id, GPU and checkpoint SHA256 per run (and per eval tag, e.g. _noflip).

Usage: python collect_results.py results/cluster_runs --prefix R1 --out results/tables/improve_val400
"""
import argparse
import csv
import glob
import json
import os
import os.path as osp
import re


def fmt(v, scale=100.0):
    return '' if v is None else f'{scale * v:.2f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--prefix', default='R1')
    ap.add_argument('--split', default='val400')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rows = []
    for rd in sorted(glob.glob(osp.join(args.root, args.prefix + '*'))):
        rid = osp.basename(rd)
        rj = json.load(open(osp.join(rd, 'run.json'))) if osp.isfile(osp.join(rd, 'run.json')) else {}
        for mp in sorted(glob.glob(osp.join(rd, f'metrics_{args.split}*.json'))):
            tag = re.match(rf'metrics_{args.split}(.*)\.json$', osp.basename(mp)).group(1)
            if tag.endswith('_mmpose'):
                continue
            m = json.load(open(mp))
            mm = m['metrics']
            apt = m.get('ap_per_oks_thr', {}).get('paper', {}) or {}
            rows.append(dict(
                run_id=rid, eval_tag=tag or '-', AP=fmt(mm.get('AP')), AP50=fmt(mm.get('AP50')),
                AP75=fmt(mm.get('AP75')), AR=fmt(mm.get('AR')), AP_80_95=fmt(mm.get('AP_80_95')),
                AP_at_90=fmt(apt.get('0.90')), AP_at_95=fmt(apt.get('0.95')),
                best_epoch=rj.get('best_epoch'), max_epochs=rj.get('max_epochs'),
                lr=rj.get('lr'), seed=rj.get('seed'), job=rj.get('slurm_job_id'),
                gpu=rj.get('gpu'), ckpt=rj.get('checkpoint'),
                ckpt_sha256=rj.get('checkpoint_sha256'), git_tree=(rj.get('git_tree') or '')[:12]))
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    if not rows:
        print('no runs found')
        return
    with open(args.out + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = ['run_id', 'eval_tag', 'AP', 'AP50', 'AP75', 'AR', 'AP_80_95', 'AP_at_90', 'AP_at_95',
            'best_epoch', 'max_epochs', 'lr', 'seed', 'job', 'gpu']
    with open(args.out + '.md', 'w') as f:
        f.write(f'Split: {args.split}. Metrics: eval/eval_coco.py `metrics` (paper protocol, rescored), x100.\n\n')
        f.write('| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n')
        for r in rows:
            f.write('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n')
    print(open(args.out + '.md').read())


if __name__ == '__main__':
    main()
