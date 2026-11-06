#!/usr/bin/env python
"""Check that every image referenced by the SyRIP splits of a config exists under its data_prefix,
and print split sizes / annotation counts. Usage: python check_data.py CONFIG"""
import json
import os.path as osp
import sys

from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
dsets = list(cfg.train_dataloader.dataset.datasets) + [cfg.val_dataloader.dataset,
                                                         cfg.test_dataloader.dataset]
bad = 0
for d in dsets:
    ann = osp.join(d.data_root, d.ann_file)
    js = json.load(open(ann))
    root = osp.join(d.data_root, d.data_prefix['img'])
    miss = [im['file_name'] for im in js['images'] if not osp.isfile(osp.join(root, im['file_name']))]
    syn = sum(bool(im.get('is_synthetic')) for im in js['images'])
    print(f"{d.ann_file}: {len(js['images'])} images ({syn} synthetic), "
          f"{len(js['annotations'])} anns, root={root}, missing={len(miss)} {miss[:3]}")
    bad += len(miss)
sys.exit(1 if bad else 0)
