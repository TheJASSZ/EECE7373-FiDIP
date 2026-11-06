#!/usr/bin/env python3
"""Print results/registry.csv rows for repro retrain runs from their run dirs (run.json, epochs.csv,
metrics_*.json from eval/eval_coco.py, metrics_upstream_*.txt). val400_AP / test100_AP = eval_coco 'paper'
protocol AP of the best-Val400 checkpoint (fallback: upstream-printed); final-epoch, 'fixed'-protocol and
upstream-printed values go to the notes.
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


def ec(path, proto='metrics'):
    if not os.path.isfile(path):
        return None
    m = json.load(open(path)).get(proto)
    return 100 * m['AP'] if m else None


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
        J = lambda n, pr='metrics': ec(os.path.join(d, 'metrics_%s.json' % n), pr)
        U = lambda n: ap(os.path.join(d, 'metrics_upstream_%s.txt' % n))
        t_best = J('test100_best') if J('test100_best') is not None else U('test100_best')
        v_best = J('val400_best') if J('val400_best') is not None else 100 * float(best['AP'])
        f = lambda v: '' if v is None else '%.2f' % v
        notes = ('eval_coco paper protocol (upstream-printed in parentheses); lambda=%s seed=%s; best Val400 epoch %s, '
                 'final epoch %s: Val400 AP %s (%s); Test100 final %s (%s); Test100 fixed protocol best/final %s/%s; '
                 'train %ss on %s' % (
                     rj.get('lambda'), rj.get('seed'), best['epoch'], E[-1]['epoch'], f(J('val400_final')),
                     f(100 * float(E[-1]['AP'])), f(J('test100_final')) or 'n/a (Val400-only run)', f(U('test100_final')),
                     f(J('test100_best', 'metrics_fixed')), f(J('test100_final', 'metrics_fixed')), rj.get('train_seconds'),
                     (rj.get('gpu') or ['?'])[0].split(',')[0]))
        variant = re.sub(r'_s\d+$', '', rid.split('_', 1)[1])
        w.writerow([rid, rj['slurm'].get('job_id'), 'repro', 'DarkPose+FiDIP HRNet-W48 384x288 (retrain)', variant,
                    rj.get('seed'), 'infant yaml + scripts/train_fidip.sh', sha, rj['slurm'].get('partition'),
                    (rj.get('gpu') or ['?'])[0].split(',')[0], 'done', f(v_best), f(t_best),
                    rj['checkpoints'].get('best_val400', {}).get('path', ''),
                    rj['checkpoints'].get('best_val400', {}).get('sha256', ''), notes])
    sys.stdout.write(out.getvalue())


if __name__ == '__main__':
    main()
