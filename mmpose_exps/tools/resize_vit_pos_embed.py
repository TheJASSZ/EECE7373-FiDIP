#!/usr/bin/env python
"""Resize the absolute position embedding of an MMPose ViTPose checkpoint to a new patch grid.

Why: the COCO ViTPose checkpoints are trained at 256x192 (patch grid 16x12 = 192 tokens with
patch 16 + padding 2). To fine-tune at 384x288 (grid 24x18 = 432 tokens) the pos_embed must be
interpolated. mmpretrain 1.2.0 tries to do this on load (VisionTransformer._prepare_pos_embed),
but it infers the source grid as int(sqrt(L)) x int(sqrt(L)) -- 13x13 for L=192 -- and fails its
own `L == src_h * src_w` assertion. We do the same bicubic interpolation (align_corners=False,
mmpretrain.models.utils.resize_pos_embed semantics) with the true (H, W) grid, offline, and save a
new checkpoint whose SHA256 is recorded. Leading extra tokens (cls slot), if present, are kept
unchanged; mmpretrain strips them on load exactly as it does for the original checkpoint.

Usage: python resize_vit_pos_embed.py SRC.pth DST.pth [--src-grid 16 12] [--dst-grid 24 18]
"""
import argparse
import hashlib
import json
import os

import torch
import torch.nn.functional as F


def sha256(path, bs=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(bs), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--src-grid', type=int, nargs=2, default=[16, 12], metavar=('H', 'W'))
    ap.add_argument('--dst-grid', type=int, nargs=2, default=[24, 18], metavar=('H', 'W'))
    ap.add_argument('--key', default='backbone.pos_embed')
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    pe = sd[args.key]
    sh, sw = args.src_grid
    dh, dw = args.dst_grid
    n_extra = pe.shape[1] - sh * sw
    assert pe.ndim == 3 and n_extra in (0, 1), f'unexpected {args.key} shape {tuple(pe.shape)}'
    extra, grid = pe[:, :n_extra], pe[:, n_extra:]
    c = grid.shape[-1]
    grid = grid.reshape(1, sh, sw, c).permute(0, 3, 1, 2).float()
    grid = F.interpolate(grid, size=(dh, dw), mode='bicubic', align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, dh * dw, c).to(pe.dtype)
    sd[args.key] = torch.cat([extra, grid], dim=1).contiguous()

    note = dict(pos_embed_resized_from=list(pe.shape), pos_embed_resized_to=list(sd[args.key].shape),
                src_grid_hw=[sh, sw], dst_grid_hw=[dh, dw], mode='bicubic', align_corners=False,
                n_extra_tokens=n_extra, src_file=os.path.basename(args.src), src_sha256=sha256(args.src))
    if isinstance(ckpt, dict) and 'meta' in ckpt and isinstance(ckpt['meta'], dict):
        ckpt['meta']['fidip_improve_note'] = note
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(ckpt, args.dst)
    note['dst_file'] = os.path.basename(args.dst)
    note['dst_sha256'] = sha256(args.dst)
    with open(args.dst + '.json', 'w') as f:
        json.dump(note, f, indent=1)
    print(json.dumps(note, indent=1))


if __name__ == '__main__':
    main()
