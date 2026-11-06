#!/usr/bin/env python3
"""Print results/registry.csv rows for repro retrain runs from their run dirs (run.json, epochs.csv,
metrics_upstream_*). val400_AP = best-Val400-epoch AP from epochs.csv (in-training validation, upstream
printed); test100_AP = upstream-printed AP of the best-Val400 checkpoint (final-epoch values in notes).
  python3 -I scripts/registry_rows_repro.py $R/runs/R010_* ...
"""
import csv
import io
import json
import os
import re
import sys


def ap(path):
    if not os.path.isfile(path):
        return None
    m = re.search(r'Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s+all \| maxDets= 20 \] = ([0-9.]+)',
                  open(path).read())
    return 100 * float(m.group(1)) if m else None


def main():
    out = io.StringIO()
    w = csv.writer(out)
    for d in sys.argv[1:]:
        rj = json.load(open(os.path.join(d, 'run.json')))
        E = list(csv.DictReader(open(os.path.join(d, 'epochs.csv'))))
        best = max(E, key=lambda r: (float(r['AP']), -int(r['epoch'])))
        rid = rj['run_id']
        g = rj.get('git_pushed', {})
        sha = '%s/tree:%s%s' % (g.get('head', '?')[:12], g.get('tree', '?')[:12], '+dirty' if g.get('dirty_any') else '')
        t_best, t_final = ap(os.path.join(d, 'metrics_upstream_test100_best.txt')), ap(os.path.join(d, 'metrics_upstream_test100_final.txt'))
        f = lambda v: '' if v is None else '%.1f' % v
        notes = ('upstream-printed; lambda=%s seed=%s; best Val400 epoch %s (AP %.1f), final epoch %s Val400 AP %.1f; '
                 'Test100 final-epoch AP %s; train %ss on %s' % (
                     rj.get('lambda'), rj.get('seed'), best['epoch'], 100 * float(best['AP']), E[-1]['epoch'],
                     100 * float(E[-1]['AP']), f(t_final) or 'n/a (Val400-only run)', rj.get('train_seconds'),
                     (rj.get('gpu') or ['?'])[0].split(',')[0]))
        variant = re.sub(r'_s\d+$', '', rid.split('_', 1)[1])
        w.writerow([rid, rj['slurm'].get('job_id'), 'repro', 'DarkPose+FiDIP HRNet-W48 384x288 (retrain)', variant,
                    rj.get('seed'), 'infant yaml + scripts/train_fidip.sh', sha, rj['slurm'].get('partition'),
                    (rj.get('gpu') or ['?'])[0].split(',')[0], 'done', '%.1f' % (100 * float(best['AP'])), f(t_best),
                    rj['checkpoints'].get('best_val400', {}).get('path', ''),
                    rj['checkpoints'].get('best_val400', {}).get('sha256', ''), notes])
    sys.stdout.write(out.getvalue())


if __name__ == '__main__':
    main()
