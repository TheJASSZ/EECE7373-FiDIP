#!/usr/bin/env python
"""Evaluate one MMPose checkpoint on Val400 or Test100 (GT boxes) and dump its predictions.

Outputs in --out-dir:
  preds_<split><tag>.json          COCO results list (input to eval/eval_coco.py)
  metrics_<split><tag>_mmpose.json MMPose CocoMetric numbers -- bookkeeping only. MMPose's
                                   xtcocotools has no id-0 artifact, so on Test100 its AP equals
                                   eval_coco.py's `fixed` protocol, NOT the paper-comparable one.
Every reported number comes from eval/eval_coco.py applied to preds_<split><tag>.json.

Guard: --split test100 is refused unless the run id appears in a table row of
results/PREREGISTRATION.md (the copy synced to the cluster), per CLAUDE.md.

Usage: python eval_ckpt.py CONFIG CKPT --split val400 --out-dir RUN_DIR [--tag _noflip]
       [--no-flip] [--run-id RID] [--cfg-options k=v ...]
"""
import argparse
import json
import os
import os.path as osp
import shutil
import time

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

SPLITS = {'val400': 'splits/val400.json', 'test100': 'splits/test100.json'}
PREREG = '/scratch/purohitham.v/fidip/repo/results/PREREGISTRATION.md'


def preregistered(run_id, path=PREREG):
    if not osp.isfile(path):
        return False
    with open(path) as f:
        rows = [ln for ln in f if ln.lstrip().startswith('|')]
    return any(run_id in ln for ln in rows[2:])  # skip header + separator rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--split', choices=sorted(SPLITS), required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--no-flip', action='store_true')
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--prereg', default=PREREG)
    ap.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = ap.parse_args()

    run_id = args.run_id or osp.basename(osp.normpath(args.out_dir))
    if args.split == 'test100' and not preregistered(run_id, args.prereg):
        raise SystemExit(f'REFUSED: {run_id} has no row in {args.prereg}; '
                         'Test100 is only for pre-registered configs.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    ann = SPLITS[args.split]
    cfg.test_dataloader.dataset.ann_file = ann
    cfg.test_dataloader.dataset.data_prefix = dict(img=cfg.img_val)
    cfg.test_dataloader.dataset.test_mode = True
    cfg.test_dataloader.dataset.pop('bbox_file', None)      # GT boxes
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = osp.join(args.out_dir, f'mmpose_{args.split}{args.tag}')
    cfg.test_evaluator = dict(type='CocoMetric', ann_file=cfg.data_root + ann,
                              outfile_prefix=prefix)
    if args.no_flip:
        cfg.model.test_cfg.flip_test = False
    cfg.load_from = args.checkpoint
    cfg.resume = False
    cfg.work_dir = osp.join(args.out_dir, f'eval_{args.split}{args.tag}')

    t0 = time.time()
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    pred_path = osp.join(args.out_dir, f'preds_{args.split}{args.tag}.json')
    shutil.move(prefix + '.keypoints.json', pred_path)
    out = dict(
        note='MMPose CocoMetric (xtcocotools, no id-0 artifact). Bookkeeping only; reported '
             'numbers come from eval/eval_coco.py on ' + osp.basename(pred_path),
        run_id=run_id, split=args.split, ann_file=cfg.data_root + ann,
        checkpoint=osp.abspath(args.checkpoint), config=osp.abspath(args.config),
        flip_test=bool(cfg.model.test_cfg.get('flip_test', False)),
        cfg_options=args.cfg_options, seconds=round(time.time() - t0, 1),
        metrics={k: float(v) for k, v in metrics.items()})
    with open(osp.join(args.out_dir, f'metrics_{args.split}{args.tag}_mmpose.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
