#!/usr/bin/env python3
"""Summarize repro retrains: per run (best-Val400 epoch, its Val400 AP, final-epoch Val400 AP, Test100 AP of
both checkpoints from the upstream-printed metrics files) and mean +- sd per variant over seeds.

  python3 -I scripts/summarize_retrains.py $R/runs R010 R011 ... [--csv out.csv]
Run ids may be prefixes (R010 matches R010_fidip-w48_lambda5e-4_s1).
"""
import csv
import glob
import os
import re
import statistics as st
import sys


def upstream_ap(path):
    """AP (IoU=0.50:0.95, all) from a metrics_upstream_*.txt file, in percent."""
    if not os.path.isfile(path):
        return None
    for line in open(path):
        m = re.search(r'Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s+all \| maxDets= 20 \] = ([0-9.]+)', line)
        if m:
            return 100 * float(m.group(1))
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out_csv = sys.argv[sys.argv.index('--csv') + 1] if '--csv' in sys.argv else ''
    root, prefixes = args[0], [a for a in args[1:] if a != out_csv]
    rows = []
    for p in prefixes:
        for d in sorted(glob.glob(os.path.join(root, p + '*'))):
            ep = os.path.join(d, 'epochs.csv')
            if not os.path.isfile(ep):
                continue
            E = list(csv.DictReader(open(ep)))
            best = max(E, key=lambda r: (float(r['AP']), -int(r['epoch'])))
            rid = os.path.basename(d)
            variant = re.sub(r'_s\d+$', '', rid.split('_', 1)[1]) if '_' in rid else rid
            rows.append({'run_id': rid, 'variant': variant, 'epochs': len(E), 'best_epoch': int(best['epoch']),
                         'val400_AP_best': 100 * float(best['AP']), 'val400_AP_final': 100 * float(E[-1]['AP']),
                         'test100_AP_best': upstream_ap(os.path.join(d, 'metrics_upstream_test100_best.txt')),
                         'test100_AP_final': upstream_ap(os.path.join(d, 'metrics_upstream_test100_final.txt')),
                         'train_mse_final': float(E[-1]['train_mse']), 'val_loss_final': float(E[-1]['val_loss'])})
    keys = ['run_id', 'variant', 'epochs', 'best_epoch', 'val400_AP_best', 'val400_AP_final', 'test100_AP_best',
            'test100_AP_final', 'train_mse_final', 'val_loss_final']
    fmt = lambda v: '' if v is None else (('%.2e' % v if abs(v) < 0.01 else '%.2f' % v) if isinstance(v, float) else str(v))
    print(' | '.join(keys))
    for r in rows:
        print(' | '.join(fmt(r[k]) for k in keys))
    print('\nvariant | n | val400_AP_best | val400_AP_final | test100_AP_best | test100_AP_final  (mean +- sd)')
    for v in sorted({r['variant'] for r in rows}):
        g = [r for r in rows if r['variant'] == v]
        cell = []
        for k in ('val400_AP_best', 'val400_AP_final', 'test100_AP_best', 'test100_AP_final'):
            xs = [r[k] for r in g if r[k] is not None]
            cell.append('%.2f +- %.2f' % (st.mean(xs), st.stdev(xs)) if len(xs) > 1 else (fmt(xs[0]) if xs else ''))
        print('%s | %d | %s' % (v, len(g), ' | '.join(cell)))
    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
