#!/usr/bin/env python
"""visualize.py -- qualitative figures: GT vs prediction skeleton overlays on the SyRIP images.

Run on the cluster (images live there) inside an sbatch job:
  python eval/visualize.py gallery  --per-image PI.json --images IMG_DIR --label "FiDIP" --slot 1 --out FIG_BASE
  python eval/visualize.py failures --per-image PI.json --images IMG_DIR --label "FiDIP" --slot 1 --out FIG_BASE
  python eval/visualize.py compare  --a PI_A.json --b PI_B.json --label-a A --label-b B --slot-a 1 --slot-b 2 \
                                    --images IMG_DIR --out FIG_BASE

Encoding (identity never colour-only):
- GT: white skeleton with hollow markers.
- Prediction: the model's categorical colour with filled markers.
- Marker shape = body side (circle = left joints, square = right joints, diamond = nose), so a
  left/right swap shows as circles sitting on the GT's right limbs.
- Each panel title gives the file name, OKS and the automatic taxonomy tags.

Images that cannot be found are drawn on a grey canvas, so the code also runs on mock data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_coco as ev  # noqa: E402
import plot_style as ps  # noqa: E402
import taxonomy as tx  # noqa: E402

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402

LEFT = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT = {2, 4, 6, 8, 10, 12, 14, 16}
PAD = 0.12
MAX_SIDE = 450


def _load_records(path):
    with open(os.path.expandvars(path)) as f:
        d = json.load(f)
    return d["records"]


def find_image(images_dir, file_name, _cache={}):
    if not images_dir or not file_name:
        return None
    p = os.path.join(images_dir, file_name)
    if os.path.exists(p):
        return p
    if images_dir not in _cache:
        idx = {}
        for root, _, files in os.walk(images_dir):
            for fn in files:
                idx.setdefault(fn, os.path.join(root, fn))
        _cache[images_dir] = idx
    return _cache[images_dir].get(os.path.basename(file_name))


def crop_for(rec, images_dir):
    """Square crop centred on the GT box (side = max(w, h) * (1 + 2*PAD)), grey-padded outside the
    image, so every panel has the same aspect. Returns (rgb array, (x0, y0) offset, scale, found)."""
    from PIL import Image
    x, y, w, h = [float(v) for v in rec["bbox"]]
    side = max(w, h) * (1 + 2 * PAD)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    S = max(1, int(round(side)))
    path = find_image(images_dir, rec.get("file_name"))
    canvas = Image.new("RGB", (S, S), (200, 199, 195))
    if path:
        im = Image.open(path).convert("RGB")
        W, H = im.size
        box = (max(0, x0), max(0, y0), min(W, x0 + S), min(H, y0 + S))
        if box[2] > box[0] and box[3] > box[1]:
            canvas.paste(im.crop(box), (box[0] - x0, box[1] - y0))
    s = min(1.0, MAX_SIDE / S)
    if s < 1.0:
        canvas = canvas.resize((max(1, int(S * s)), max(1, int(S * s))))
    return np.asarray(canvas), (x0, y0), s, path is not None


def draw_skeleton(ax, kpts, off, s, color, gt=False, vis=None):
    k = np.asarray(kpts, dtype=float)
    xy = (k[:, :2] - off) * s
    lab = (np.asarray(vis) > 0) if vis is not None else np.ones(len(k), bool)
    lw, halo = (0.9, 2.0) if gt else (1.3, 2.6)
    for i, j in ev.SKELETON:
        if lab[i] and lab[j]:
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], color="black", lw=halo, alpha=0.45, solid_capstyle="round",
                    zorder=3 if gt else 5)
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=color, lw=lw, solid_capstyle="round", zorder=4 if gt else 6)
    for idx in range(len(k)):
        if not lab[idx]:
            continue
        m = "o" if idx in LEFT else ("s" if idx in RIGHT else "D")
        if gt:
            occluded = vis is not None and vis[idx] == 1
            ax.plot(*xy[idx], marker=m, ms=4.2, mfc="none", mec="white", mew=0.9, zorder=7, alpha=0.95,
                    ls="none")
            if occluded:
                ax.plot(*xy[idx], marker="x", ms=2.6, color="white", mew=0.7, zorder=7, ls="none")
        else:
            ax.plot(*xy[idx], marker=m, ms=3.4, mfc=color, mec="black", mew=0.4, zorder=8, ls="none")


def tags(rec):
    a = tx.annotate(rec)
    t = []
    if a["n_swapped_pairs"]:
        t.append("swap " + "+".join(a["swapped_pairs"]))
    if a["inplane"] == "sideways":
        t.append("sideways")
    if a["inplane"] == "inverted":
        t.append("upside-down")
    if a["view"] in ("back", "profile"):
        t.append(a["view"])
    if a["n_v1"]:
        t.append(f"occl {a['n_v1']}")
    return a, t


def panel(ax, rec, images_dir, color, title_extra="", title=None):
    img, off, s, real = crop_for(rec, images_dir)
    ax.imshow(img, interpolation="bilinear")
    g = np.asarray(rec["gt_kpts"], dtype=float)
    draw_skeleton(ax, g, off, s, "white", gt=True, vis=g[:, 2])
    if rec.get("pred_kpts") is not None:
        draw_skeleton(ax, rec["pred_kpts"], off, s, color, gt=False, vis=g[:, 2] if False else None)
    a, t = tags(rec)
    name = os.path.splitext(str(rec.get("file_name")))[0]
    if title is None:
        title = f"{name}  OKS {rec['oks']:.3f}{title_extra}\n" + (", ".join(t) if t else " ")
    ax.set_title(title, fontsize=5.6, loc="left", color=ps.INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    return real


def _legend(fig, color, label):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#888", marker="o", mfc="none", mec="#333", ls="-", lw=0.9, label="ground truth"),
         Line2D([], [], color=color, marker="o", mfc=color, mec="black", ls="-", lw=1.3, label=label),
         Line2D([], [], color="none", marker="o", mfc="#666", mec="black", label="left joints"),
         Line2D([], [], color="none", marker="s", mfc="#666", mec="black", label="right joints"),
         Line2D([], [], color="none", marker="x", mec="#333", label="GT v=1 (occluded)")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=6.3, bbox_to_anchor=(0.5, 0.0))


def cmd_gallery(a):
    recs = sorted(_load_records(a.per_image), key=lambda r: r["oks"])
    n = a.n
    mid = len(recs) // 2
    rows = [("best", recs[::-1][:n]), ("median", recs[max(0, mid - n // 2): mid - n // 2 + n]), ("worst", recs[:n])]
    fig, axes = plt.subplots(3, n, figsize=(ps.FULL_W, ps.FULL_W * 3 / n * 1.12))
    col = ps.color(a.slot)
    n_real = 0
    for r, (name, rr) in enumerate(rows):
        for c in range(n):
            ax = axes[r][c]
            if c < len(rr):
                n_real += panel(ax, rr[c], a.images, col)
            else:
                ax.axis("off")
        axes[r][0].set_ylabel(name, fontsize=8, color=ps.INK, rotation=90, labelpad=4)
        axes[r][0].yaxis.set_visible(True)
    _legend(fig, col, a.label)
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.6, w_pad=0.4)
    paths = ps.save(fig, a.out)
    print("\n".join(paths) + f"\n({n_real} panels with real images)")


def cmd_failures(a):
    recs = _load_records(a.per_image)
    ann = {r["image_id"]: tx.annotate(r) for r in recs}
    q1 = np.percentile([x["sqrt_area"] for x in ann.values()], 100 / 3)
    cats = [("L/R swap", lambda r, x: x["n_swapped_pairs"] > 0),
            ("occlusion (≥2 v=1)", lambda r, x: x["n_v1"] >= 2),
            ("unusual orientation", lambda r, x: x["inplane"] in ("sideways", "inverted") or x["view"] in
             ("back", "profile")),
            ("small scale", lambda r, x: x["sqrt_area"] <= q1)]
    n = a.n
    fig, axes = plt.subplots(len(cats), n, figsize=(ps.FULL_W, ps.FULL_W * len(cats) / n * 1.12))
    col = ps.color(a.slot)
    summary = {}
    shown = set()
    for r, (name, fn) in enumerate(cats):
        allsel = sorted([x for x in recs if fn(x, ann[x["image_id"]])], key=lambda x: x["oks"])
        sel = [x for x in allsel if x["image_id"] not in shown] or allsel   # distinct examples per row
        shown.update(x["image_id"] for x in sel[:n])
        summary[name] = {"n": len(allsel), "mean_oks": float(np.mean([x["oks"] for x in allsel])) if allsel else None,
                         "shown": [x["file_name"] for x in sel[:n]]}
        for c in range(n):
            ax = axes[r][c]
            if c < len(sel):
                panel(ax, sel[c], a.images, col)
            else:
                ax.axis("off")
        axes[r][0].set_ylabel(f"{name}\n(n={len(allsel)})", fontsize=7, color=ps.INK)
        axes[r][0].yaxis.set_visible(True)
    _legend(fig, col, a.label)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=0.6, w_pad=0.4)
    paths = ps.save(fig, a.out)
    with open(a.out + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(paths))


def cmd_compare(a):
    ra = {(r["image_id"], r["ann_id"]): r for r in _load_records(a.a)}
    rb = {(r["image_id"], r["ann_id"]): r for r in _load_records(a.b)}
    keys = sorted(set(ra) & set(rb))
    d = sorted(keys, key=lambda k: rb[k]["oks"] - ra[k]["oks"])
    gains, losses = d[::-1][:a.n], d[:a.n]
    fig, axes = plt.subplots(2, 2 * a.n, figsize=(ps.FULL_W, ps.FULL_W * 2 / (2 * a.n) * 1.25))
    ca, cb = ps.color(a.slot_a), ps.color(a.slot_b)
    def gt_tags(rec):
        x = tx.annotate(rec)
        t = [v for v, c in (("sideways", x["inplane"] == "sideways"), ("upside-down", x["inplane"] == "inverted"),
                            (x["view"], x["view"] in ("back", "profile")),
                            (f"occl {x['n_v1']}", x["n_v1"] > 0)) if c]
        return ", ".join(t), x

    def swap_tag(rec):
        x = tx.annotate(rec)
        return ("swap " + "+".join(x["swapped_pairs"])) if x["n_swapped_pairs"] else " "

    for r, (name, ks) in enumerate([("gains", gains), ("losses", losses)]):
        for i, k in enumerate(ks):
            delta = rb[k]["oks"] - ra[k]["oks"]
            stem = os.path.splitext(str(ra[k].get("file_name")))[0]
            gtt, _ = gt_tags(ra[k])
            panel(axes[r][2 * i], ra[k], a.images, ca,
                  title=f"{stem}: {gtt or 'no tags'}\n{a.label_a} {ra[k]['oks']:.3f}  {swap_tag(ra[k])}")
            panel(axes[r][2 * i + 1], rb[k], a.images, cb,
                  title=f"Δ OKS {delta:+.3f}\n{a.label_b} {rb[k]['oks']:.3f}  {swap_tag(rb[k])}")
        axes[r][0].set_ylabel(f"largest {name}\nof {a.label_b}", fontsize=6.5, color=ps.INK)
        axes[r][0].yaxis.set_visible(True)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#888", marker="o", mfc="none", mec="#333", lw=0.9, label="ground truth"),
         Line2D([], [], color=ca, marker="o", mfc=ca, mec="black", lw=1.3, label=a.label_a),
         Line2D([], [], color=cb, marker="o", mfc=cb, mec="black", lw=1.3, label=a.label_b),
         Line2D([], [], color="none", marker="o", mfc="#666", mec="black", label="left"),
         Line2D([], [], color="none", marker="s", mfc="#666", mec="black", label="right")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=6.3)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.5, w_pad=0.3)
    print("\n".join(ps.save(fig, a.out)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gallery", "failures"):
        s = sub.add_parser(name)
        s.add_argument("--per-image", required=True)
        s.add_argument("--images", default=None)
        s.add_argument("--label", default="prediction")
        s.add_argument("--slot", type=int, default=1)
        s.add_argument("--n", type=int, default=4)
        s.add_argument("--out", required=True)
    s = sub.add_parser("compare")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--label-a", default="A")
    s.add_argument("--label-b", default="B")
    s.add_argument("--slot-a", type=int, default=1)
    s.add_argument("--slot-b", type=int, default=2)
    s.add_argument("--images", default=None)
    s.add_argument("--n", type=int, default=2)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"gallery": cmd_gallery, "failures": cmd_failures, "compare": cmd_compare}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
