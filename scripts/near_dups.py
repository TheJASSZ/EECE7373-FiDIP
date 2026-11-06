#!/usr/bin/env python3
"""Near-duplicate image check across SyRIP splits (perceptual dHash, flip-aware).

Exact duplicates are found by md5 in make_splits.py. This catches re-encoded / resized / mirrored
copies: 64-bit difference hash of the 9x8 grayscale thumbnail, Hamming distance <= --thr counts as a
candidate. Each candidate pair is drawn side by side in a contact sheet for visual confirmation.

Usage (fidip env, compute node):
  python scripts/near_dups.py --raw $R/data/raw --out $R/data/splits/near_dups --thr 8
"""
import argparse
import hashlib
import json
import os

import cv2
import numpy as np


def dhash(gray):
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(''.join('1' if b else '0' for b in bits), 2)


SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0


def pose_oks(ga, ia, gb, ib):
    """OKS between two GT poses in image-normalized coordinates (x/W, y/H), so that a re-scaled copy of the
    same frame scores ~1; scale from a's normalized bbox area; keypoints visible in both."""
    ka, kb = np.array(ga['keypoints'], float).reshape(17, 3), np.array(gb['keypoints'], float).reshape(17, 3)
    va = (ka[:, 2] > 0) & (kb[:, 2] > 0)
    if not va.any():
        return 0.0
    na, nb = ka[:, :2] / [ia['width'], ia['height']], kb[:, :2] / [ib['width'], ib['height']]
    area = (ga['bbox'][2] / ia['width']) * (ga['bbox'][3] / ia['height'])
    d2 = ((na - nb) ** 2).sum(1)
    return float(np.exp(-d2 / (2 * area * (2 * SIGMAS) ** 2 + np.spacing(1)))[va].mean())


def hashes(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return dhash(img), dhash(img[:, ::-1]), hashlib.md5(open(path, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--thr', type=int, default=8)
    ap.add_argument('--oks', type=float, default=0.85, help='pose-OKS threshold for a near-duplicate')
    ap.add_argument('--exclude-out', default='', help='write Val400 exclusion list here')
    a = ap.parse_args()
    S = os.path.join(a.raw, 'SyRIP')
    os.makedirs(a.out, exist_ok=True)
    v100 = json.load(open(os.path.join(S, 'annotations/validate100/person_keypoints_validate_infant.json')))
    v500 = json.load(open(os.path.join(S, 'annotations/validate500/person_keypoints_validate_infant.json')))
    tr = json.load(open(os.path.join(S, 'annotations/200R_1000S/person_keypoints_train_infant.json')))
    test = {i['id'] for i in v100['images']}
    items = []  # (split, id, path)
    gt = {}     # (split, id) -> (annotation, image info); Test100 uses validate100's own GT
    a100 = {x['image_id']: x for x in v100['annotations']}
    a500 = {x['image_id']: x for x in v500['annotations']}
    atr = {x['image_id']: x for x in tr['annotations']}
    for i in v500['images']:
        sp = 'test100' if i['id'] in test else 'val400'
        items.append((sp, i['id'], os.path.join(S, 'images/validate_infant', i['file_name'])))
        gt[(sp, i['id'])] = ((a100 if sp == 'test100' else a500)[i['id']], i)
    for i in tr['images']:
        if not i['is_synthetic']:
            items.append(('train_real', i['id'], os.path.join(S, 'images/train_infant', i['file_name'])))
            gt[('train_real', i['id'])] = (atr[i['id']], i)
    H = [hashes(p) for _, _, p in items]
    pairs = []
    for x in range(len(items)):
        for y in range(x + 1, len(items)):
            sx, sy = items[x][0], items[y][0]
            if sx == sy == 'train_real':
                continue
            d = min(bin(H[x][0] ^ H[y][0]).count('1'), bin(H[x][0] ^ H[y][1]).count('1'))
            if d <= a.thr:
                pairs.append({'a': [sx, items[x][1]], 'b': [sy, items[y][1]], 'dhash_dist': d,
                              'exact_md5': H[x][2] == H[y][2],
                              'mirrored': bin(H[x][0] ^ H[y][1]).count('1') < bin(H[x][0] ^ H[y][0]).count('1'),
                              'pose_oks': round(pose_oks(*gt[(sx, items[x][1])], *gt[(sy, items[y][1])]), 4)})
    pairs.sort(key=lambda p: p['dhash_dist'])
    json.dump(pairs, open(os.path.join(a.out, 'near_dups.json'), 'w'), indent=1)
    # contact sheets, one per pair type, one row per pair (2 thumbnails of height 200)
    def sheet(sel, fname):
        rows = []
        for p in sel:
            ims = []
            for split, iid in (p['a'], p['b']):
                path = next(pp for s_, i, pp in items if s_ == split and i == iid)
                im = cv2.imread(path)
                im = cv2.resize(im, (int(im.shape[1] * 200 / im.shape[0]), 200))
                cv2.putText(im, '%s %d' % (split, iid), (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ims.append(im)
            row = np.concatenate([ims[0], np.full((200, 8, 3), 255, np.uint8), ims[1]], 1)
            cv2.putText(row, 'd=%d md5=%s oks=%.2f' % (p['dhash_dist'], p['exact_md5'], p['pose_oks']), (4, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            rows.append(row)
        if rows:
            W = max(r.shape[1] for r in rows)
            for part in range(0, len(rows), 6):   # 6 pairs per image, viewable at full resolution
                chunk = rows[part:part + 6]
                out = np.concatenate([np.pad(r, ((0, 6), (0, W - r.shape[1]), (0, 0)), constant_values=255)
                                      for r in chunk], 0)
                cv2.imwrite(os.path.join(a.out, fname.replace('.jpg', '_part%d.jpg' % (part // 6))), out,
                            [cv2.IMWRITE_JPEG_QUALITY, 75])
    kinds = {'val400-test100': {'val400', 'test100'}, 'test100-train': {'test100', 'train_real'},
             'val400-train': {'val400', 'train_real'}}
    for kname, kset in kinds.items():
        sel = [p for p in pairs if {p['a'][0], p['b'][0]} == kset]
        sheet(sel[:40], 'sheet_%s.jpg' % kname)
        print('%s: %d candidate pairs (d<=%d)' % (kname, len(sel), a.thr))
    vt = [p for p in pairs if {p['a'][0], p['b'][0]} == {'val400', 'test100'}]
    excl = {}
    for p in vt:
        if p['exact_md5'] or p['pose_oks'] >= a.oks:
            v = p['a'] if p['a'][0] == 'val400' else p['b']
            t = p['b'] if p['a'][0] == 'val400' else p['a']
            excl.setdefault(v[1], []).append('test%d dhash=%d md5=%s pose_oks=%.3f' % (
                t[1], p['dhash_dist'], p['exact_md5'], p['pose_oks']))
    print('Val400 exclusions (exact md5, or dHash<=%d and pose OKS>=%.2f vs a Test100 image):' % (a.thr, a.oks))
    for k in sorted(excl):
        print('  val %d <- %s' % (k, '; '.join(excl[k])))
    if a.exclude_out:
        with open(a.exclude_out, 'w') as f:
            f.write('# Val400 image ids excluded as (near-)duplicates of Test100 images (scripts/near_dups.py:\n'
                    '# exact md5, or dHash<=%d and GT pose OKS>=%.2f in normalized coordinates); visually checked.\n'
                    % (a.thr, a.oks))
            for k in sorted(excl):
                f.write('%d  # %s\n' % (k, '; '.join(excl[k])))
    for kname, kset in kinds.items():
        sel = sorted([p for p in pairs if {p['a'][0], p['b'][0]} == kset], key=lambda p: -p['pose_oks'])
        print('%s: top pose-OKS pairs %s' % (kname, [(p['a'][1], p['b'][1], p['dhash_dist'], p['pose_oks']) for p in sel[:12]]))
    cross = [p for p in pairs if {p['a'][0], p['b'][0]} in ({'val400', 'test100'}, {'val400', 'train_real'},
                                                            {'test100', 'train_real'})]
    print('n images', len(items), 'candidate pairs (d<=%d)' % a.thr, len(pairs), 'cross-split', len(cross))



if __name__ == '__main__':
    main()
