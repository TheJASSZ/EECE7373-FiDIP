#!/usr/bin/env python
"""Split the official SyRIP train_infant GT (200 real + 1000 synthetic) by images[].is_synthetic.

Writes <out_dir>/train_real200.json and <out_dir>/train_syn1000.json (COCO format, same
categories/info, annotations of the kept images only) plus a manifest with counts and SHA256.
Used by the 1:1 real:synthetic mix (CombinedDataset, sample_ratio_factor=[5, 1]) and by the
real-only ablation. Nothing about Val400/Test100 is read here.

Usage: python make_train_subsets.py $R/data/splits/train_infant.json $R/data/improve_splits
"""
import argparse
import hashlib
import json
import os


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('train_json')
    ap.add_argument('out_dir')
    ap.add_argument('--expect-real', type=int, default=200)
    ap.add_argument('--expect-syn', type=int, default=1000)
    args = ap.parse_args()

    with open(args.train_json) as f:
        gt = json.load(f)
    imgs = gt['images']
    missing = [im['id'] for im in imgs if 'is_synthetic' not in im]
    assert not missing, f'{len(missing)} images lack is_synthetic, e.g. {missing[:5]}'

    def is_syn(im):
        v = im['is_synthetic']
        return v if isinstance(v, bool) else bool(int(v))

    groups = {'real200': [im for im in imgs if not is_syn(im)],
              'syn1000': [im for im in imgs if is_syn(im)]}
    assert len(groups['real200']) == args.expect_real, len(groups['real200'])
    assert len(groups['syn1000']) == args.expect_syn, len(groups['syn1000'])

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {'source': os.path.abspath(args.train_json), 'source_sha256': sha256(args.train_json)}
    for name, keep in groups.items():
        ids = {im['id'] for im in keep}
        anns = [a for a in gt['annotations'] if a['image_id'] in ids]
        out = {k: v for k, v in gt.items() if k not in ('images', 'annotations')}
        out['images'] = keep
        out['annotations'] = anns
        path = os.path.join(args.out_dir, f'train_{name}.json')
        with open(path, 'w') as f:
            json.dump(out, f)
        manifest[name] = {'path': path, 'n_images': len(keep), 'n_annotations': len(anns),
                          'sha256': sha256(path),
                          'file_names_head': sorted(im['file_name'] for im in keep)[:3]}
        assert len(anns) >= len(keep), f'{name}: fewer annotations than images'
    with open(os.path.join(args.out_dir, 'MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
