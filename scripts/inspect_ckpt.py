#!/usr/bin/env python3
"""Check that checkpoints load strictly into the upstream FiDIP pose nets (CPU only).

For each (yaml, checkpoint): build model_p with the upstream code, report the checkpoint's key count,
'module.' prefix, missing/unexpected keys against model_p.state_dict(), shape mismatches, and whether a
strict load succeeds. Run from the upstream root:
  cd third_party/Infant-Pose-Estimation && python ../../scripts/inspect_ckpt.py CFG CKPT [CFG CKPT ...]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))
from config import cfg  # noqa: E402
import models  # noqa: E402,F401


def check(cfg_file, ckpt):
    c = cfg.clone()
    c.defrost()
    c.merge_from_file(cfg_file)
    c.freeze()
    model_p, model_d = eval('models.' + c.MODEL.NAME + '.get_adaptive_pose_net')(c, is_train=False)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    kind = type(sd).__name__
    if isinstance(sd, dict) and 'state_dict' in sd and not torch.is_tensor(sd.get('state_dict')):
        print('  (training checkpoint dict with keys %s)' % list(sd.keys())[:8])
        sd = sd['state_dict']
    prefixed = len(sd) and all(k.startswith('module.') for k in sd)
    if prefixed:
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    want = model_p.state_dict()
    missing = sorted(set(want) - set(sd))
    unexpected = sorted(set(sd) - set(want))
    shape = sorted(k for k in set(want) & set(sd) if tuple(want[k].shape) != tuple(sd[k].shape))
    print('%s\n  cfg %s | %s with %d tensors, module-prefix=%s | model_p has %d | missing %d unexpected %d '
          'shape-mismatch %d' % (ckpt, cfg_file, kind, len(sd), bool(prefixed), len(want), len(missing),
                                 len(unexpected), len(shape)))
    for lbl, lst in (('missing', missing), ('unexpected', unexpected), ('shape', shape)):
        if lst:
            print('   %s: %s%s' % (lbl, lst[:6], ' ...' if len(lst) > 6 else ''))
    try:
        model_p.load_state_dict(sd, strict=True)
        print('  STRICT LOAD OK')
    except RuntimeError as e:
        print('  STRICT LOAD FAILED: %s' % str(e).splitlines()[0])
    n_params = sum(p.numel() for p in model_p.parameters())
    print('  model_p params: %.2fM; first frozen-boundary param (idx 108): %s' % (
        n_params / 1e6, [n for i, (n, _) in enumerate(model_p.named_parameters()) if i in (108, 109)]))


if __name__ == '__main__':
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        check(args[i], args[i + 1])
