#!/usr/bin/env python
"""GPU sanity check for the MMPose env: versions, CUDA arch support, mmcv CUDA ops, and, per
config, checkpoint key matching plus a timed train step (forward + backward + AdamW step) at the
training batch size with peak memory. Writes a JSON report.

Usage: python verify_env.py --out report.json CONFIG[:BATCH] [CONFIG[:BATCH] ...]
"""
import argparse
import json
import os
import platform
import socket
import time

import torch


def mmcv_ops_check():
    from mmcv.ops import nms, roi_align
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                         dtype=torch.float32, device='cuda')
    scores = torch.tensor([0.9, 0.8, 0.7], device='cuda')
    _, keep = nms(boxes, scores, 0.5)
    feat = torch.randn(1, 8, 32, 32, device='cuda')
    rois = torch.tensor([[0, 0, 0, 16, 16]], dtype=torch.float32, device='cuda')
    out = roi_align(feat, rois, (7, 7), 1.0, 0, 'avg', True)
    torch.cuda.synchronize()
    return dict(nms_keep=keep.tolist(), roi_align_shape=list(out.shape))


def check_config(cfg_path, batch, n_warm=2, n_iter=5):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.utils import import_modules_from_strings
    from mmpose.registry import MODELS
    cfg = Config.fromfile(cfg_path)
    init_default_scope('mmpose')
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    model = MODELS.build(cfg.model)
    rep = dict(config=cfg_path, load_from=cfg.load_from, batch=batch)
    ckpt = torch.load(cfg.load_from, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    res = model.load_state_dict(sd, strict=False)
    rep['missing_keys'] = list(res.missing_keys)
    rep['unexpected_keys'] = list(res.unexpected_keys)
    rep['n_params_M'] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    model = model.cuda().train()
    w, h = cfg.codec['input_size']
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        for i in range(n_warm + n_iter):
            x = torch.randn(batch, 3, h, w, device='cuda')
            torch.cuda.synchronize()
            t0 = time.time()
            hm = model.head.forward(model.extract_feat(x))
            loss = hm.float().pow(2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= n_warm:
                times.append(time.time() - t0)
        rep['heatmap_shape'] = list(hm.shape)
        rep['train_step_s'] = round(sum(times) / len(times), 4)
        rep['peak_mem_GB'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        rep['status'] = 'ok'
    except torch.cuda.OutOfMemoryError as e:
        rep['status'] = 'OOM'
        rep['error'] = str(e).splitlines()[0]
    model.eval()
    with torch.no_grad():
        x = torch.randn(min(batch, 32), 3, h, w, device='cuda')
        torch.cuda.synchronize()
        t0 = time.time()
        model.head.forward(model.extract_feat(x))
        torch.cuda.synchronize()
        rep['infer_32_s'] = round(time.time() - t0, 4)
    del model, opt
    torch.cuda.empty_cache()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('configs', nargs='*')
    args = ap.parse_args()
    import numpy, torchvision, mmcv, mmengine, mmpose, mmpretrain, xtcocotools, pycocotools  # noqa
    rep = dict(
        host=socket.gethostname(), python=platform.python_version(),
        slurm_job=os.environ.get('SLURM_JOB_ID'),
        versions=dict(numpy=numpy.__version__, torch=torch.__version__,
                      torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      torchvision=torchvision.__version__, mmcv=mmcv.__version__,
                      mmengine=mmengine.__version__, mmpose=mmpose.__version__,
                      mmpretrain=mmpretrain.__version__),
        cuda_available=torch.cuda.is_available(),
        device=torch.cuda.get_device_name(0),
        capability='sm_%d%d' % torch.cuda.get_device_capability(0),
        arch_list=torch.cuda.get_arch_list())
    try:
        rep['mmcv_ops'] = mmcv_ops_check()
    except Exception as e:  # report, do not abort: pose top-down does not use mmcv CUDA ops
        rep['mmcv_ops'] = f'FAILED: {type(e).__name__}: {e}'
    rep['configs'] = []
    for item in args.configs:
        path, _, b = item.partition(':')
        try:
            rep['configs'].append(check_config(path, int(b or 32)))
        except Exception as e:
            rep['configs'].append(dict(config=path, status=f'FAILED: {type(e).__name__}: {e}'))
        print(json.dumps(rep['configs'][-1]), flush=True)
    with open(args.out, 'w') as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'configs'}, indent=1))


if __name__ == '__main__':
    main()
