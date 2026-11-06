#!/usr/bin/env python3
"""Verify the extracted SyRIP trees against the downloaded zips (stdlib only).

For every zip member: the file exists on disk under the extraction root, has the same size and the same
CRC32 as recorded in the zip central directory; no extra files exist on disk. In addition, for a random
sample of members, the md5 of the decompressed member (`unzip -p` equivalent) equals the md5 of the file
on disk.

Usage: python3 -I scripts/verify_raw.py --downloads $R/data/downloads --raw $R/data/raw [--sample 50]
"""
import argparse
import hashlib
import os
import random
import sys
import zipfile
import zlib

ZIPS = {  # zip file -> directory the zip's top-level folder was extracted into
    'SyRIP.zip': '',                       # members SyRIP/...           -> $RAW/SyRIP/...
    'syrip_for_train.zip': 'syrip_for_train',  # members syrip/...       -> $RAW/syrip_for_train/syrip/...
}


def crc32(path):
    c = 0
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            c = zlib.crc32(chunk, c)
    return c & 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--downloads', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--sample', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    ok = True
    for zname, sub in ZIPS.items():
        base = os.path.join(a.raw, sub) if sub else a.raw
        with zipfile.ZipFile(os.path.join(a.downloads, zname)) as z:
            members = [m for m in z.infolist() if not m.is_dir()]
            tops = {m.filename.split('/')[0] for m in members}
            missing, bad_size, bad_crc = [], [], []
            for m in members:
                p = os.path.join(base, m.filename)
                if not os.path.isfile(p):
                    missing.append(m.filename)
                elif os.path.getsize(p) != m.file_size:
                    bad_size.append(m.filename)
                elif crc32(p) != m.CRC:
                    bad_crc.append(m.filename)
            on_disk = set()
            for top in tops:
                for root, _, files in os.walk(os.path.join(base, top)):
                    for f in files:
                        on_disk.add(os.path.relpath(os.path.join(root, f), base))
            extra = sorted(on_disk - {m.filename for m in members})
            sample = rng.sample(members, min(a.sample, len(members)))
            md5_bad = []
            for m in sample:
                h_zip = hashlib.md5(z.read(m)).hexdigest()
                h_disk = hashlib.md5(open(os.path.join(base, m.filename), 'rb').read()).hexdigest()
                if h_zip != h_disk:
                    md5_bad.append(m.filename)
            print('%s -> %s/%s: %d members | missing %d, size mismatch %d, CRC32 mismatch %d, extra on disk %d | '
                  'md5(unzip -p) == md5(disk) for %d/%d sampled members' % (
                      zname, base, '|'.join(sorted(tops)), len(members), len(missing), len(bad_size), len(bad_crc),
                      len(extra), len(sample) - len(md5_bad), len(sample)))
            for lbl, lst in (('missing', missing), ('size', bad_size), ('crc', bad_crc), ('extra', extra),
                             ('md5', md5_bad)):
                if lst:
                    ok = False
                    print('  %s: %s' % (lbl, lst[:10]))
    print('RAW TREES VERIFIED' if ok else 'RAW TREES DIFFER FROM ZIPS')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
