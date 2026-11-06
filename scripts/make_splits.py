#!/usr/bin/env python3
"""Build the framework-neutral SyRIP splits (COCO keypoint GT) and verify them.

Writes (OUT = $R/data/splits):
  train_infant.json  official SyRIP train (200 real + 1000 synthetic) = SyRIP.zip annotations/200R_1000S,
                     area := bbox_w*bbox_h (D8); everything else as shipped.
  test100.json       official validate100 (= the paper's Test100), byte-identical copy (D8, D10).
  val400.json        validate500 minus the validate100 image ids, area := bbox_w*bbox_h (D8); bbox,
                     keypoints, ids, image order as shipped in validate500.
  validate500.json   validate500 with area := bbox_w*bbox_h (D8).  NOTE: its 100 Test100 images carry
                     validate500's own (different) boxes; evaluate Test100 only with test100.json.
  VERIFY.json/.txt   the verification report (counts, overlaps by id/file name/md5, area stats, ...).

and, in the upstream layout (SYRIP = $R/data/syrip, = DATASET.ROOT of the FiDIP code):
  annotations/person_keypoints_{val400,test100,validate500}.json -> symlinks to the files above
  images/{val400,test100}/test<id>.jpg -> per-file symlinks into raw SyRIP images/validate_infant
  images/validate500 -> symlink to raw SyRIP images/validate_infant

Image resolution rule for every split JSON: <SYRIP>/images/<split>/<file_name>
(file_name is the shipped basename, e.g. test27.jpg, train00101.jpg).

Decisions applied: D1 (Val400/Test100 protocol), D8 (area fix), D10 (ids exactly as shipped).
Stdlib only, so it runs with any python3 >= 3.8 (run it inside a Slurm job: it hashes ~3.6k images).
"""
import argparse
import collections
import hashlib
import json
import os
import shutil
import statistics
import struct
import sys

TEST100_SRC = 'annotations/validate100/person_keypoints_validate_infant.json'
VAL500_SRC = 'annotations/validate500/person_keypoints_validate_infant.json'
TRAIN_SRC = 'annotations/200R_1000S/person_keypoints_train_infant.json'
FT_TRAIN = 'annotations/person_keypoints_train_infant.json'          # in syrip_for_train/syrip (NOT used)
FT_PRE = 'annotations/person_keypoints_train_pre_infant.json'        # in syrip_for_train/syrip
FT_VAL = 'annotations/person_keypoints_validate_infant.json'         # in syrip_for_train/syrip


def load(p):
    with open(p) as f:
        return json.load(f)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


NON_JPEG = []   # files whose bytes are not JPEG (e.g. validate test489.jpg is a PNG); cv2/PIL decode by content


def jpeg_wh(p):
    """(width, height) from the JPEG SOF marker or the PNG IHDR chunk (no PIL needed)."""
    with open(p, 'rb') as f:
        d = f.read(1 << 17)
    if d[:8] == b'\x89PNG\r\n\x1a\n':
        if p not in NON_JPEG:
            NON_JPEG.append(p)
        return struct.unpack('>II', d[16:24])
    assert d[:2] == b'\xff\xd8', 'neither JPEG nor PNG: %s' % p
    i = 2
    while i + 9 < len(d):
        if d[i] != 0xFF:
            i += 1
            continue
        m = d[i + 1]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack('>HH', d[i + 5:i + 9])
            return w, h
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        i += 2 + struct.unpack('>H', d[i + 2:i + 4])[0]
    raise ValueError('no SOF marker in %s' % p)


def fix_area(d):
    """D8: area := bbox_w * bbox_h. Returns a new dict; ids, order, bbox, keypoints untouched."""
    out = dict(d)
    out['annotations'] = []
    for a in d['annotations']:
        a = dict(a)
        a['area'] = float(a['bbox'][2]) * float(a['bbox'][3])
        out['annotations'].append(a)
    return out


def area_stats(d):
    ar = [a['area'] for a in d['annotations']]
    return {'n': len(ar), 'n_unique': len(set(ar)), 'min': min(ar), 'median': statistics.median(ar),
            'max': max(ar),
            'n_area_eq_wh_1e-3': sum(abs(a['area'] - a['bbox'][2] * a['bbox'][3]) <= 1e-3
                                     for a in d['annotations'])}


def assert_area_ok(name, d):
    st = area_stats(d)
    assert st['n_unique'] > 1, '%s: constant area (placeholder) %r' % (name, st)
    assert st['n_area_eq_wh_1e-3'] == st['n'], '%s: area != w*h for some annotations %r' % (name, st)
    return st


def assert_ids_as_shipped(name, d, src, img_subset=None):
    """D10: image ids, annotation ids (and their image_id links), bbox and keypoints equal the source."""
    keep = set(i['id'] for i in src['images']) if img_subset is None else set(img_subset)
    s_imgs = [i for i in src['images'] if i['id'] in keep]
    s_anns = [a for a in src['annotations'] if a['image_id'] in keep]
    assert [i['id'] for i in d['images']] == [i['id'] for i in s_imgs], name + ': image ids/order changed'
    assert [i['file_name'] for i in d['images']] == [i['file_name'] for i in s_imgs], name + ': file_name changed'
    assert [(a['id'], a['image_id']) for a in d['annotations']] == \
        [(a['id'], a['image_id']) for a in s_anns], name + ': annotation ids changed'
    assert all(a['bbox'] == b['bbox'] and a['keypoints'] == b['keypoints']
               for a, b in zip(d['annotations'], s_anns)), name + ': bbox/keypoints changed'
    assert d['categories'] == src['categories'], name + ': categories changed'


def summary(d):
    per = collections.Counter(a['image_id'] for a in d['annotations'])
    return {
        'n_images': len(d['images']), 'n_annotations': len(d['annotations']),
        'image_id_min': min(i['id'] for i in d['images']), 'image_id_max': max(i['id'] for i in d['images']),
        'annotation_id_min': min(a['id'] for a in d['annotations']),
        'anns_per_image_hist': {str(k): v for k, v in sorted(collections.Counter(per.values()).items())},
        'n_images_without_ann': len(set(i['id'] for i in d['images']) - set(per)),
        'n_multi_person_images': sum(1 for v in per.values() if v > 1),
        'is_synthetic': {str(k): v for k, v in collections.Counter(i.get('is_synthetic') for i in d['images']).items()},
        'num_keypoints_hist': {str(k): v for k, v in sorted(collections.Counter(
            a['num_keypoints'] for a in d['annotations']).items())},
        'visibility_flags': {str(k): v for k, v in sorted(collections.Counter(
            v for a in d['annotations'] for v in a['keypoints'][2::3]).items())},
        'iscrowd': {str(k): v for k, v in collections.Counter(a['iscrowd'] for a in d['annotations']).items()},
    }


def check_images(name, d, img_dir, rep):
    """Every image exists and its JPEG size equals width/height in the JSON. Returns {id: md5}."""
    hashes, bad_wh, missing = {}, [], []
    for im in d['images']:
        p = os.path.join(img_dir, im['file_name'])
        if not os.path.isfile(p):
            missing.append(im['file_name'])
            continue
        if tuple(jpeg_wh(p)) != (im['width'], im['height']):
            bad_wh.append((im['id'], im['file_name'], jpeg_wh(p), (im['width'], im['height'])))
        hashes[im['id']] = md5(p)
    by_md5 = collections.defaultdict(list)
    for k, h in hashes.items():
        by_md5[h].append(k)
    dups = sorted(sorted(v) for v in by_md5.values() if len(v) > 1)
    rep['images'][name] = {'dir': img_dir, 'n_missing': len(missing), 'missing_examples': missing[:5],
                           'n_wh_mismatch': len(bad_wh), 'wh_mismatch_examples': bad_wh[:5],
                           'n_duplicate_md5_within_split': len(hashes) - len(by_md5),
                           'duplicate_md5_image_id_groups': dups}
    return hashes


def link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst) and os.readlink(dst) == src:
            return
        if os.path.isdir(dst) and not os.path.islink(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', required=True, help='$R/data/raw (holds SyRIP/ and syrip_for_train/syrip/)')
    ap.add_argument('--syrip', required=True, help='$R/data/syrip (upstream layout, DATASET.ROOT)')
    ap.add_argument('--out', required=True, help='$R/data/splits')
    ap.add_argument('--ids-out', required=True, help='repo splits/ dir for the committed ID lists')
    ap.add_argument('--exclude-val-ids', default='',
                    help='optional file of validate500 image ids to drop from Val400 (reviewed near-duplicates '
                         'of Test100 images, see scripts/near_dups.py); one id per line, # comments allowed')
    args = ap.parse_args()

    S = os.path.join(args.raw, 'SyRIP')
    T = os.path.join(args.raw, 'syrip_for_train', 'syrip')
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.ids_out, exist_ok=True)
    rep = {'sources': {}, 'splits': {}, 'area': {}, 'images': {}, 'overlap': {}, 'notes': []}

    # ---------------- load sources
    src = {'train_200R_1000S': os.path.join(S, TRAIN_SRC), 'validate100': os.path.join(S, TEST100_SRC),
           'validate500': os.path.join(S, VAL500_SRC), 'for_train_train_infant': os.path.join(T, FT_TRAIN),
           'for_train_train_pre_infant': os.path.join(T, FT_PRE), 'for_train_validate_infant': os.path.join(T, FT_VAL)}
    for k, p in src.items():
        rep['sources'][k] = {'path': p, 'sha256': sha256(p)}
    train_src, v100, v500 = load(src['train_200R_1000S']), load(src['validate100']), load(src['validate500'])
    for k, d in (('train_200R_1000S', train_src), ('validate100', v100), ('validate500', v500)):
        rep['area']['source:' + k] = area_stats(d)

    # ---------------- build splits
    test_ids = [i['id'] for i in v100['images']]
    v500_ids = [i['id'] for i in v500['images']]
    test_set, v500_set = set(test_ids), set(v500_ids)
    assert len(test_set) == len(test_ids) == 100 and len(v500_set) == len(v500_ids) == 500
    assert test_set <= v500_set, 'validate100 image ids are not a subset of validate500'
    fn500 = {i['id']: i['file_name'] for i in v500['images']}
    assert all(fn500[i['id']] == i['file_name'] for i in v100['images']), 'validate100/500 file_name mismatch'
    # D1: Val400 must not contain any Test100 image. Ids/file names are disjoint by construction, but some
    # validate500 images are byte-identical copies of Test100 images under another id: drop those from Val400
    # (Test100 itself is never modified), plus any reviewed near-duplicates passed via --exclude-val-ids.
    raw_val_dir = os.path.join(S, 'images', 'validate_infant')
    md5_500 = {i['id']: md5(os.path.join(raw_val_dir, i['file_name'])) for i in v500['images']}
    excl = {}
    for k in v500_ids:
        same = sorted(j for j in test_ids if md5_500[j] == md5_500[k])
        if k not in test_set and same:
            excl[k] = 'byte-identical (md5) to Test100 image id(s) %s' % same
    if args.exclude_val_ids:
        for line in open(args.exclude_val_ids):
            line = line.split('#')[0].strip()
            if line:
                k = int(line.split()[0])
                assert k in v500_set and k not in test_set, k
                excl.setdefault(k, 'near-duplicate of a Test100 image (reviewed list %s)' % args.exclude_val_ids)
    rep['val400_excluded'] = {str(k): v for k, v in sorted(excl.items())}
    val_ids = [i for i in v500_ids if i not in test_set and i not in excl]
    assert len(val_ids) == 400 - len(excl)
    val_keep = set(val_ids)

    v500_fixed = fix_area(v500)
    val400 = dict(v500_fixed)
    val400['images'] = [i for i in v500_fixed['images'] if i['id'] in val_keep]
    val400['annotations'] = [a for a in v500_fixed['annotations'] if a['image_id'] in val_keep]
    train = fix_area(train_src)

    out = {'train_infant': train, 'val400': val400, 'validate500': v500_fixed}
    for name, d in out.items():
        with open(os.path.join(args.out, name + '.json'), 'w') as f:
            json.dump(d, f)
    shutil.copyfile(src['validate100'], os.path.join(args.out, 'test100.json'))   # exact bytes (D8, D10)

    # ---------------- re-read what was written and assert (D8, D10)
    W = {n: load(os.path.join(args.out, n + '.json')) for n in ('train_infant', 'test100', 'val400', 'validate500')}
    assert sha256(os.path.join(args.out, 'test100.json')) == rep['sources']['validate100']['sha256']
    assert_ids_as_shipped('train_infant', W['train_infant'], train_src)
    assert_ids_as_shipped('test100', W['test100'], v100)
    assert_ids_as_shipped('val400', W['val400'], v500, img_subset=val_ids)
    assert_ids_as_shipped('validate500', W['validate500'], v500)
    for n, d in W.items():
        rep['area'][n] = assert_area_ok(n, d)
        rep['splits'][n] = summary(d)
        rep['splits'][n]['sha256'] = sha256(os.path.join(args.out, n + '.json'))
    # D10: the GT annotation with id 0 is kept on purpose (COCOeval's gt-id-0 artifact, as in the paper's numbers)
    assert min(a['id'] for a in W['test100']['annotations']) == 0
    ts = W['train_infant']['images']
    assert sum(1 for i in ts if i['is_synthetic']) == 1000 and sum(1 for i in ts if not i['is_synthetic']) == 200

    # ---------------- upstream layout for the new splits (DATASET.ROOT = args.syrip)
    ann_dir, img_root = os.path.join(args.syrip, 'annotations'), os.path.join(args.syrip, 'images')
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(img_root, exist_ok=True)
    for n in ('val400', 'test100', 'validate500'):
        link(os.path.join(args.out, n + '.json'), os.path.join(ann_dir, 'person_keypoints_%s.json' % n))
    link(raw_val_dir, os.path.join(img_root, 'validate500'))
    for n, d in (('val400', W['val400']), ('test100', W['test100'])):
        dd = os.path.join(img_root, n)
        if os.path.islink(dd):
            os.remove(dd)
        os.makedirs(dd, exist_ok=True)
        want = set(i['file_name'] for i in d['images'])
        for f in os.listdir(dd):                       # drop stale links
            if f not in want:
                os.remove(os.path.join(dd, f))
        for f in want:
            link(os.path.join(raw_val_dir, f), os.path.join(dd, f))

    # ---------------- image checks: existence, JPEG size == JSON size, md5
    H = {}
    H['train_infant'] = check_images('train_infant', W['train_infant'], os.path.join(img_root, 'train_infant'), rep)
    H['test100'] = check_images('test100', W['test100'], os.path.join(img_root, 'test100'), rep)
    H['val400'] = check_images('val400', W['val400'], os.path.join(img_root, 'val400'), rep)
    H['validate500'] = check_images('validate500', W['validate500'], os.path.join(img_root, 'validate500'), rep)
    H['validate_infant(upstream,100)'] = check_images('validate_infant(upstream,100)', load(
        os.path.join(ann_dir, 'person_keypoints_validate_infant.json')), os.path.join(img_root, 'validate_infant'), rep)
    for n in ('train_infant', 'test100', 'val400', 'validate500'):
        r = rep['images'][n]
        assert r['n_missing'] == 0 and r['n_wh_mismatch'] == 0, (n, r)

    # ---------------- overlaps (D1): by id, file name and md5 of image bytes
    fn = {n: {i['id']: i['file_name'] for i in W[n]['images']} for n in W}
    md5_test, md5_val, md5_train = set(H['test100'].values()), set(H['val400'].values()), set(H['train_infant'].values())
    ov = rep['overlap']
    ov['test100_subset_validate500_by_id'] = set(fn['test100']) <= set(fn['validate500'])
    ov['test100_subset_validate500_by_md5'] = md5_test <= set(H['validate500'].values())
    ov['val400_x_test100_by_id'] = len(set(fn['val400']) & set(fn['test100']))
    ov['val400_x_test100_by_file_name'] = len(set(fn['val400'].values()) & set(fn['test100'].values()))
    ov['val400_x_test100_by_md5'] = len(md5_val & md5_test)
    t_by_md5 = {h: k for k, h in H['train_infant'].items()}
    ov['train_x_val400_by_md5'] = len(md5_train & md5_val)
    ov['train_x_test100_by_md5_pairs(test_id,train_id)'] = sorted(
        (k, t_by_md5[h]) for k, h in H['test100'].items() if h in t_by_md5)
    up = H['validate_infant(upstream,100)']
    ov['upstream_validate_infant_100_ids_whose_bytes_differ_from_SyRIP_zip'] = sorted(
        k for k in up if up[k] != H['test100'][k])
    assert ov['test100_subset_validate500_by_id'] and ov['test100_subset_validate500_by_md5']
    assert ov['val400_x_test100_by_id'] == ov['val400_x_test100_by_file_name'] == ov['val400_x_test100_by_md5'] == 0
    assert ov['train_x_val400_by_md5'] == 0, 'train/Val400 leakage by identical image bytes'
    if ov['train_x_test100_by_md5_pairs(test_id,train_id)']:
        rep['notes'].append(
            'OFFICIAL-DATASET LEAK, kept as shipped for comparability with the paper: %d Test100 images are '
            'byte-identical to annotated official training images, (test id, train id) = %s. Every model trained on '
            'SyRIP train (including the published FiDIP) has seen them; report Test100 also without these ids as '
            'a sensitivity check.' % (len(ov['train_x_test100_by_md5_pairs(test_id,train_id)']),
                                      ov['train_x_test100_by_md5_pairs(test_id,train_id)']))
    if excl:
        rep['notes'].append('Val400 has %d images (not 400): dropped %s.' % (len(val_ids), rep['val400_excluded']))
    nd = os.path.join(args.out, 'near_dups', 'near_dups.json')   # written by scripts/near_dups.py (optional)
    if os.path.isfile(nd):
        near = [p for p in load(nd) if {p['a'][0], p['b'][0]} == {'test100', 'train_real'}
                and (p['exact_md5'] or p['pose_oks'] >= 0.85)]
        ov['test100_x_train_near_dups(pose_oks>=0.85 or md5)'] = sorted(
            (p['a'][1] if p['a'][0] == 'test100' else p['b'][1], p['b'][1] if p['a'][0] == 'test100' else p['a'][1],
             p['dhash_dist'], p['pose_oks']) for p in near)
        rep['notes'].append('Test100 images with an exact or near-exact (dHash<=10, pose OKS>=0.85) copy in the '
                            'official train set, (test id, train id, dHash, pose OKS): %s'
                            % ov['test100_x_train_near_dups(pose_oks>=0.85 or md5)'])
    for k in ov['upstream_validate_infant_100_ids_whose_bytes_differ_from_SyRIP_zip']:
        same_as = [j for j, h in H['test100'].items() if h == up[k]]
        rep['notes'].append(
            'DEFECT in syrip_for_train.zip images/validate_infant (the folder the upstream README layout uses for '
            'TEST_SET validate_infant): test%d.jpg has different bytes than SyRIP.zip test%d.jpg%s; '
            '(id, file, actual WxH, GT WxH) = %s. Our test100 split uses the SyRIP.zip images, which match the GT.' % (
                k, k, (' and is byte-identical to test%s.jpg' % same_as[0]) if same_as else '',
                rep['images']['validate_infant(upstream,100)']['wh_mismatch_examples']))
    # extra unannotated real images shipped in images/train_infant (informational leakage check)
    tdir = os.path.join(img_root, 'train_infant')
    annotated = set(fn['train_infant'].values())
    extra = sorted(f for f in os.listdir(tdir) if f.endswith('.jpg') and f not in annotated)
    md5_extra = {md5(os.path.join(tdir, f)) for f in extra}
    ov['train_infant_dir_unannotated_images'] = len(extra)
    ov['train_infant_dir_unannotated_x_validate500_by_md5'] = len(md5_extra & set(H['validate500'].values()))
    ov['train_infant_dir_unannotated_examples'] = extra[:3] + extra[-3:]

    # ---------------- Test100 GT: validate100 vs validate500's copy of the same 100 images
    a100 = {a['image_id']: a for a in v100['annotations']}
    a500 = {a['image_id']: a for a in v500['annotations']}
    kdiff = [k for k in test_ids if a100[k]['keypoints'] != a500[k]['keypoints']]
    bdiff = [k for k in test_ids if a100[k]['bbox'] != a500[k]['bbox']]
    rep['notes'].append('Test100 GT: validate100 vs validate500 copies of the same 100 images: '
                        'keypoints differ on %d images %s, bbox differs on %d images.' % (len(kdiff), kdiff, len(bdiff)))

    # ---------------- syrip_for_train.zip: train_pre_infant (used for PRE_EPOCH) and the defective train json
    pre = load(src['for_train_train_pre_infant'])
    rep['splits']['train_pre_infant(syrip_for_train)'] = summary(pre)
    rep['area']['train_pre_infant(syrip_for_train)'] = area_stats(pre)
    check_images('train_pre_infant(syrip_for_train)', pre, os.path.join(T, 'images', 'train_pre_infant'), rep)
    ft = load(src['for_train_train_infant'])
    ft_syn = [i for i in ft['images'] if i['is_synthetic']]
    mini = sum(1 for i in ft_syn if i.get('image_source') == 'MINI_dataset' or '/syn_' in i.get('original_file_name', ''))
    ft_rep = {}
    check_images('for_train_train_infant(NOT USED)', ft, os.path.join(T, 'images', 'train_infant'), {'images': ft_rep})
    ft_md5 = {}
    for i in ft['images']:
        ft_md5[i['id']] = md5(os.path.join(T, 'images', 'train_infant', i['file_name']))
    same_bytes = sum(ft_md5[k] == H['train_infant'][k] for k in ft_md5)
    oa = {a['image_id']: a for a in train_src['annotations']}
    ta = {a['image_id']: a for a in ft['annotations']}
    same_real = all(oa[k]['keypoints'] == ta[k]['keypoints'] and oa[k]['bbox'] == ta[k]['bbox']
                    for k in oa if not any(i['id'] == k and i['is_synthetic'] for i in ft_syn))
    rep['notes'].append(
        'DEFECT, not used: syrip_for_train.zip annotations/person_keypoints_train_infant.json. Same 1200 ids and file '
        'names as SyRIP 200R_1000S and identical image bytes (%d/1200 md5 equal), real-image labels identical (%s), but '
        '%d/1000 synthetic entries carry MINI-RGBD labels (original_file_name like "10/syn_00360.png", image_source '
        '"MINI_dataset"); JSON size != JPEG size on %d images. Training uses SyRIP.zip 200R_1000S instead.'
        % (same_bytes, same_real, mini, ft_rep['for_train_train_infant(NOT USED)']['n_wh_mismatch']))

    # ---------------- ID lists (committed to the repo)
    for n, ids in (('val400', val_ids), ('test100', test_ids)):
        with open(os.path.join(args.ids_out, '%s_image_ids.txt' % n), 'w') as f:
            f.write(''.join('%d\n' % i for i in sorted(ids)))

    rep['image_resolution_rule'] = '<SYRIP>/images/<split>/<file_name>, SYRIP=%s' % args.syrip
    rep['non_jpeg_files'] = sorted(set(NON_JPEG))
    with open(os.path.join(args.out, 'VERIFY.json'), 'w') as f:
        json.dump(rep, f, indent=1, default=str)
    lines = ['SyRIP split verification (scripts/make_splits.py)', '',
             'image dir for split X: %s/images/X/<file_name>' % args.syrip, '']
    for n, s in rep['splits'].items():
        lines.append('%-36s imgs %4d anns %4d multi-person %d synthetic %s sha256 %s' % (
            n, s['n_images'], s['n_annotations'], s['n_multi_person_images'], s['is_synthetic'], s.get('sha256', '-')[:16]))
    lines.append('')
    for n, a in rep['area'].items():
        lines.append('area %-36s n %4d unique %4d min %10.1f median %10.1f max %10.1f area==w*h %d' % (
            n, a['n'], a['n_unique'], a['min'], a['median'], a['max'], a['n_area_eq_wh_1e-3']))
    lines.append('')
    for n, r in rep['images'].items():
        lines.append('images %-36s missing %d wh_mismatch %d dup_md5 %d %s (%s)' % (
            n, r['n_missing'], r['n_wh_mismatch'], r['n_duplicate_md5_within_split'],
            ('groups(image ids) %s' % r['duplicate_md5_image_id_groups'][:10]) if r['n_duplicate_md5_within_split'] else '',
            r['dir']))
    lines.append('non-JPEG files (PNG bytes, .jpg name; decoded fine by cv2/PIL): %s' % rep['non_jpeg_files'])
    lines.append('')
    lines += ['overlap %s = %s' % kv for kv in ov.items()]
    lines.append('')
    lines += ['NOTE ' + n for n in rep['notes']]
    with open(os.path.join(args.out, 'VERIFY.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    shutil.copyfile(os.path.join(args.out, 'VERIFY.txt'), os.path.join(args.ids_out, 'VERIFY.txt'))
    print('\n'.join(lines))
    print('ALL ASSERTIONS PASSED')


if __name__ == '__main__':
    sys.exit(main())
