#!/usr/bin/env python3
"""Fetch the released FiDIP checkpoints and the COCO DarkPose HRNet-W48 checkpoint with gdown.

Lists both Google Drive folders (no download), picks only the needed files, downloads them into the
upstream layout and writes SHA256SUMS:
  $R/models/hrnet_fidip.pth          (FiDIP_models: DarkPose+FiDIP HRNet-W48 384x288)
  $R/models/mobile_fidip.pth         (FiDIP_models: Pose-MobileNet+FiDIP 224x224)
  $R/models/coco/w48_384x288.pth     (TGA_models: DarkPose COCO HRNet-W48 384x288, the fine-tuning init)
Usage (fidip env, compute node with internet): python scripts/get_models.py --out $R/models [--list-only]
"""
import argparse
import hashlib
import json
import os
import sys

import gdown

FIDIP = 'https://drive.google.com/drive/folders/108P-1SnTqaj3xNtjYZ1o7T8z6UvUYuiC'
TGA = 'https://drive.google.com/drive/folders/14kAA1zXuKODYgrRiQmKnVcipbY7RedVV'


def norm(s):
    return s.replace('×', 'x').lower()


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--list-only', action='store_true')
    a = ap.parse_args()
    listing = {}
    for name, url in (('FiDIP_models', FIDIP), ('TGA_models', TGA)):
        files = gdown.download_folder(url=url, skip_download=True, quiet=True, use_cookies=False)
        listing[name] = [{'id': f.id, 'path': f.path} for f in files]
        print('== %s (%d files)' % (name, len(files)))
        for f in files:
            print('   %s  https://drive.google.com/uc?id=%s' % (f.path, f.id))
    os.makedirs(a.out, exist_ok=True)
    json.dump(listing, open(os.path.join(a.out, 'drive_listing.json'), 'w'), indent=1)
    if a.list_only:
        return 0

    want = {}
    for f in listing['FiDIP_models']:
        b = norm(os.path.basename(f['path']))
        if b in ('hrnet_fidip.pth', 'mobile_fidip.pth'):
            want[b] = f
    coco = [f for f in listing['TGA_models'] if norm(os.path.basename(f['path'])) == 'w48_384x288.pth']
    coco_pref = [f for f in coco if 'coco' in norm(f['path'])] or coco
    if coco_pref:
        want['coco/w48_384x288.pth'] = coco_pref[0]
    missing = {'hrnet_fidip.pth', 'mobile_fidip.pth', 'coco/w48_384x288.pth'} - set(want)
    if missing:
        print('NOT FOUND in the Drive listings:', sorted(missing))
    sums = []
    for target, f in sorted(want.items()):
        dst = os.path.join(a.out, target)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.isfile(dst):
            print('downloading %s -> %s' % (f['path'], dst))
            gdown.download(id=f['id'], output=dst + '.part', quiet=False, use_cookies=False)
            os.replace(dst + '.part', dst)
        sums.append('%s  %s  (drive id %s, drive path %s)' % (sha256(dst), target, f['id'], f['path']))
        print(sums[-1])
    with open(os.path.join(a.out, 'SHA256SUMS'), 'w') as fh:
        fh.write(''.join('%s  %s\n' % tuple(s.split('  ')[:2]) for s in sums))
    with open(os.path.join(a.out, 'PROVENANCE.txt'), 'w') as fh:
        fh.write('\n'.join(sums) + '\n')
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
