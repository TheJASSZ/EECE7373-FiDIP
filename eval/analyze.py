#!/usr/bin/env python
"""analyze.py -- quantitative analysis figures + tables from eval_coco.py per-image outputs.

Works without the images. visualize.py draws the image overlays.

  python eval/analyze.py models  --manifest M.json --outdir results/figures/test100 --tables results/tables
  python eval/analyze.py splits  --label "FiDIP released" --a test100=per_image_test100.json \
                                 --b val400=per_image_val400.json --outdir ... --tables ...
  python eval/analyze.py curves  --runs RUN_DIR [RUN_DIR ...] --name fidip_retrain --outdir ... --tables ...
  python eval/analyze.py gtbox   --pred-v100box P1.json --pred-v500box P2.json --gt-v100 test100.json \
                                 --gt-v500 validate500.json --tables ...

Manifest (models):
{"split": "test100", "gt": "<path>",
 "models": [{"key": "fidip_rel", "label": "FiDIP (released)", "slot": 1,
             "runs": [{"run_id": "R001_...", "per_image": "<per_image.json>", "metrics": "<metrics.json>"}]}, ...],
 "focus": ["fidip_rel", ...],          # <= 3 keys drawn in colour where all series can meet
 "pairs": [["fidip_rel", "vitpose_h"]],  # per-image OKS scatter comparisons
 "published": {"label": "FiDIP (paper)", "AP": 0.936, "AP50": 0.985, "AP75": 0.985, "AR": 0.946}}

Paths may use $R and other environment variables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import OrderedDict, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_coco as ev  # noqa: E402
import plot_style as ps  # noqa: E402
import taxonomy as tx  # noqa: E402

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402

JOINT_LABELS = ["nose", "L eye", "R eye", "L ear", "R ear", "L shoulder", "R shoulder", "L elbow", "R elbow",
                "L wrist", "R wrist", "L hip", "R hip", "L knee", "R knee", "L ankle", "R ankle"]
THRS = [f"{t:.2f}" for t in np.linspace(0.5, 0.95, 10)]
RNG_SEED = 0
SPLIT_COLORS = ["#3d3c39", "#a8a69f"]   # splits are not models: neutral dark/light so slot colours stay models
B_CI = 2000


# ----------------------------------------------------------------------------- io
def _p(x):
    return os.path.expandvars(os.path.expanduser(x)) if isinstance(x, str) else x


def load_per_image(path):
    with open(_p(path)) as f:
        d = json.load(f)
    return d["records"], d.get("meta", {})


def load_metrics(path):
    if not path or not os.path.exists(_p(path)):
        return None
    with open(_p(path)) as f:
        return json.load(f)


def write_csv(path, rows, header=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        return path
    header = header or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmtv(r.get(k)) for k in header})
    return path


def _fmtv(v):
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    return "" if v is None else v


def boot_mean_ci(x, B=B_CI, seed=RNG_SEED):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    if x.size == 1:
        return (float(x[0]), float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, (B, x.size))].mean(1)
    return (float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


# ----------------------------------------------------------------------------- model container
class Model:
    """One configuration: one or more runs (seeds) on one split, aligned by (image_id, ann_id)."""

    def __init__(self, spec):
        self.key = spec["key"]
        self.label = spec.get("label", self.key)
        self.slot = spec.get("slot")
        self.runs = spec["runs"]
        self.run_ids = [r.get("run_id") for r in self.runs]
        self.records = []
        self.metrics = []
        for r in self.runs:
            recs, meta = load_per_image(r["per_image"])
            self.records.append({(x["image_id"], x["ann_id"]): x for x in recs})
            mpath = r.get("metrics") or r["per_image"].replace("per_image_", "metrics_")
            self.metrics.append(load_metrics(mpath))
        keys = [set(d) for d in self.records]
        self.keys = sorted(set.intersection(*keys))
        if any(len(k) != len(self.keys) for k in keys):
            print(f"WARNING {self.key}: runs cover different instances; using the {len(self.keys)} common ones")

    def rec(self, key, run=0):
        return self.records[run][key]

    def oks(self, key):
        return float(np.mean([d[key]["oks"] for d in self.records]))

    def kpt_err(self, key):
        """(17,) mean over runs of normalised error (nan where unlabelled; inf -> nan)."""
        arr = []
        for d in self.records:
            e = np.array([np.nan if v is None else v for v in d[key]["kpt_err_norm"]], dtype=float)
            arr.append(e)
        return np.nanmean(np.stack(arr), axis=0) if len(arr) > 1 else arr[0]

    def per_thr(self, protocol="paper"):
        """(n_runs, 10) AP per OKS threshold."""
        rows = []
        for m in self.metrics:
            if m and m.get("ap_per_oks_thr", {}).get(protocol):
                rows.append([m["ap_per_oks_thr"][protocol][t] for t in THRS])
        return np.array(rows, dtype=float) if rows else None

    def metric(self, name, protocol="paper"):
        key = "metrics" if protocol == "paper" else "metrics_fixed"
        vals = [m[key][name] for m in self.metrics if m and m.get(key) and m[key].get(name) is not None]
        return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0) if vals else (None, None)

    @property
    def color(self):
        return ps.color(self.slot)


def _style(m, focus):
    infocus = (not focus) or (m.key in focus)
    return dict(color=m.color if infocus else ps.CONTEXT_GREY, marker=ps.marker(m.slot),
                lw=1.5 if infocus else 0.9, alpha=1.0 if infocus else 0.85, zorder=3 if infocus else 2)


# ----------------------------------------------------------------------------- figures
def fig_ap_vs_threshold(models, focus, published, out):
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.5), sharey=True)
    x = np.linspace(0.5, 0.95, 10)
    rows = []
    from matplotlib.patches import Patch
    for ax, proto, title in [(axes[0], "paper", "paper protocol (as published)"),
                             (axes[1], "fixed", "fixed protocol (id-0 artefact removed)")]:
        if proto == "paper":
            ax.axhspan(98.02, 99.01, color=ps.GRID, lw=0, zorder=1)
        for m in models:
            pt = m.per_thr(proto)
            if pt is None:
                continue
            st = _style(m, focus)
            mean = 100 * pt.mean(0)
            ax.plot(x, mean, color=st["color"], marker=st["marker"], ms=3.5, lw=st["lw"], zorder=st["zorder"],
                    label=m.label + (f" (n={len(pt)})" if len(pt) > 1 else ""))
            if len(pt) > 1:
                ax.fill_between(x, 100 * pt.min(0), 100 * pt.max(0), color=st["color"], alpha=0.15, lw=0)
            for t, v, lo, hi in zip(THRS, mean, 100 * pt.min(0), 100 * pt.max(0)):
                rows.append({"model": m.label, "key": m.key, "run_ids": m.run_ids, "protocol": proto,
                             "oks_thr": t, "AP_mean": v, "AP_min": lo, "AP_max": hi, "n_runs": len(pt)})
        if published and proto == "paper":
            ax.plot([0.5, 0.75], [100 * published["AP50"], 100 * published["AP75"]], ls="none", marker="D",
                    mfc="white", mec=ps.INK, ms=4.5, zorder=5, label=published.get("label", "paper"))
            strict = (10 * published["AP"] - 6 * published["AP50"]) / 4
            ax.plot([0.8, 0.95], [100 * strict] * 2, color=ps.INK, lw=1.0, zorder=5,
                    label=f"paper: implied mean AP over .80-.95 ({100 * strict:.1f})")
        ax.set_title(title, loc="left", color=ps.INK)
        ax.set_xlabel("OKS threshold")
        ax.set_xticks(x[::1])
        ax.set_xticklabels([f"{t:.2f}".lstrip("0") for t in x], fontsize=6.5)
    axes[0].set_ylabel("AP at threshold (%)")
    lo = min([l.get_ydata().min() for a in axes for l in a.get_lines() if len(l.get_ydata())] + [60])
    axes[0].set_ylim(max(0, math.floor(lo / 5) * 5 - 2), 101)
    h, l = axes[0].get_legend_handles_labels()
    h.append(Patch(color=ps.GRID, lw=0))
    l.append("paper-protocol ceiling for a perfect model")
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return ps.save(fig, out), rows


def fig_oks_distribution(models, focus, out):
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.4))
    allo = np.concatenate([[m.oks(k) for k in m.keys] for m in models])
    x0 = min(0.75, max(0.0, math.floor((np.percentile(allo, 1) - 0.02) * 20) / 20))
    bins = np.linspace(x0, 1.0, int(round((1 - x0) / 0.02)) + 1)
    t = np.linspace(x0, 1.0, 201)
    for m in models:
        st = _style(m, focus)
        o = np.array([m.oks(k) for k in m.keys])
        if (not focus) or m.key in focus:
            axes[0].hist(np.clip(o, x0, 1.0), bins=bins, histtype="step", color=st["color"], lw=1.3,
                         label=m.label)
        surv = [(o >= x).mean() * 100 for x in t]
        axes[1].plot(t, surv, color=st["color"], lw=st["lw"], zorder=st["zorder"], label=m.label)
    axes[0].set_xlabel(f"per-image OKS (values below {x0:.2f} shown at {x0:.2f})")
    axes[0].set_ylabel("images")
    axes[0].set_title("OKS histogram", loc="left")
    axes[1].set_xlabel("OKS threshold t")
    axes[1].set_ylabel("images with OKS ≥ t (%)")
    axes[1].set_title("survival curve (≈ recall at threshold t)", loc="left")
    axes[1].set_xlim(x0, 1.0)
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    return ps.save(fig, out)


def fig_per_keypoint(models, focus, out):
    show = [m for m in models if (not focus) or m.key in focus][:3]
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 3.3), sharey=True)
    y = np.arange(17)[::-1]
    h = 0.8 / max(1, len(show))
    rows = []
    for i, m in enumerate(show):
        errs = np.stack([m.kpt_err(k) for k in m.keys])      # (N, 17)
        mean, lo, hi, pck = [], [], [], []
        for j in range(17):
            mu, a, b = boot_mean_ci(errs[:, j])
            mean.append(mu)
            lo.append(a)
            hi.append(b)
            col = errs[:, j][np.isfinite(errs[:, j])]
            pck.append(100 * np.mean(col < 0.05) if col.size else np.nan)
            rows.append({"model": m.label, "key": m.key, "run_ids": m.run_ids, "joint": ev.COCO_KPT_NAMES[j],
                         "n": int(col.size), "mean_err_norm": mu, "ci_lo": a, "ci_hi": b,
                         "pck@0.05": pck[-1] / 100, "pck@0.1": float(np.mean(col < 0.1)) if col.size else None,
                         "pck@0.2": float(np.mean(col < 0.2)) if col.size else None})
        yy = y + (len(show) - 1) / 2 * h - i * h
        mean, lo, hi = np.array(mean), np.array(lo), np.array(hi)
        axes[0].barh(yy, mean, height=h * 0.9, color=m.color, label=m.label)
        axes[0].errorbar(mean, yy, xerr=[mean - lo, hi - mean], fmt="none", ecolor=ps.INK2, elinewidth=0.6,
                         capsize=0)
        axes[1].barh(yy, pck, height=h * 0.9, color=m.color, label=m.label)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(JOINT_LABELS)
    axes[0].set_xlabel("mean error / sqrt(GT box area)   (95% CI)")
    axes[1].set_xlabel("PCK@0.05 (%)")
    axes[0].set_title("per-keypoint localisation error", loc="left")
    axes[1].set_title("strict accuracy per keypoint", loc="left")
    lo_pck = np.nanmin([b.get_width() for b in axes[1].patches] + [100])
    axes[1].set_xlim(max(0, math.floor(lo_pck / 10) * 10 - 5), 100)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    return ps.save(fig, out), rows


POSTURES = ["supine", "prone", "sitting", "standing"]


def load_posture(path):
    """image_id -> posture label from a COCO json with images[].posture (SyRIP syrip_for_train
    validate_infant.json carries it for the 100 Test100 images)."""
    if not path:
        return {}
    p = _p(path)
    if not os.path.exists(p):
        print(f"WARNING: posture file {p} not found")
        return {}
    d = json.load(open(p))
    return {int(im["id"]): im.get("posture") for im in d.get("images", []) if im.get("posture")}


def taxonomy_table(models, posture=None):
    """Image-level taxonomy annotations (GT-derived, model-independent) + per-model OKS/swaps."""
    base = models[0]
    ann = {}
    for k in base.keys:
        ann[k] = tx.annotate(base.rec(k))
        ann[k]["posture"] = (posture or {}).get(k[0], "unknown")
    s = np.array([ann[k]["sqrt_area"] for k in base.keys])
    q1, q2 = np.percentile(s, [100 / 3, 200 / 3])
    for k in base.keys:
        v = ann[k]["sqrt_area"]
        ann[k]["scale_bin"] = "small" if v <= q1 else ("medium" if v <= q2 else "large")
        n1 = ann[k]["n_v1"]
        ann[k]["occ_bin"] = "0" if n1 == 0 else ("1" if n1 == 1 else "≥2")
    per_model = {}
    for m in models:
        d = {}
        for k in m.keys:
            sw = [tx.annotate(m.rec(k, r)) for r in range(len(m.runs))]
            d[k] = {"oks": m.oks(k), "err": float(np.nanmean(m.kpt_err(k))),
                    "n_swapped_pairs": float(np.mean([a["n_swapped_pairs"] for a in sw])),
                    "swapped_pairs": sorted(set(p for a in sw for p in a["swapped_pairs"])),
                    "swap_any": float(np.mean([a["n_swapped_pairs"] > 0 for a in sw])),
                    "full_swap_gain": float(np.mean([a["full_swap_gain"] or 0 for a in sw]))}
        per_model[m.key] = d
    return ann, per_model, (q1, q2)


FACTORS = [("inplane", "in-plane rotation", ["upright", "sideways", "inverted"]),
           ("view", "view (front≈supine)", ["front", "back", "profile"]),
           ("occ_bin", "occluded joints (v=1)", ["0", "1", "≥2"]),
           ("scale_bin", "scale tertile", ["small", "medium", "large"])]
POSTURE_FACTOR = ("posture", "posture (GT label)", POSTURES)


def factors_for(ann):
    """Use the GT posture labels instead of the geometric view proxy when every image has one."""
    if ann and all(v.get("posture") in POSTURES for v in ann.values()):
        return [FACTORS[0], POSTURE_FACTOR, FACTORS[2], FACTORS[3]], True
    return list(FACTORS), False


def fig_taxonomy(models, focus, ann, per_model, out):
    show = [m for m in models if (not focus) or m.key in focus][:3]
    fig, grid = plt.subplots(2, 3, figsize=(ps.FULL_W, 3.9), sharey=True)
    axes = list(grid.ravel()[:5])
    leg_ax = grid.ravel()[5]
    rows = []
    w = 0.8 / max(1, len(show))
    for ax, (fk, title, cats) in zip(axes, factors_for(ann)[0]):
        for i, m in enumerate(show):
            for c, cat in enumerate(cats):
                ks = [k for k in m.keys if ann.get(k, {}).get(fk) == cat]
                vals = [1 - per_model[m.key][k]["oks"] for k in ks]
                mu, lo, hi = boot_mean_ci(vals)
                xx = c + (i - (len(show) - 1) / 2) * w
                if ks:
                    ax.bar(xx, 100 * mu, width=w * 0.9, color=m.color, label=m.label if c == 0 else None)
                    ax.errorbar(xx, 100 * mu, yerr=[[100 * (mu - lo)], [100 * (hi - mu)]], fmt="none",
                                ecolor=ps.INK2, elinewidth=0.6)
                rows.append({"model": m.label, "key": m.key, "factor": fk, "category": cat, "n_images": len(ks),
                             "mean_1_minus_oks": mu, "ci_lo": lo, "ci_hi": hi,
                             "frac_oks_lt_0.85": float(np.mean([per_model[m.key][k]["oks"] < 0.85 for k in ks]))
                             if ks else None,
                             "deficit_share": float(sum(vals) / max(1e-12, sum(1 - per_model[m.key][k]["oks"]
                                                                                for k in m.keys)))})
        ns = [sum(1 for k in show[0].keys if ann.get(k, {}).get(fk) == cat) for cat in cats]
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([f"{c}\nn={n}" for c, n in zip(cats, ns)], fontsize=6.5)
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.grid(axis="x", visible=False)
    ax = axes[-1]
    for i, m in enumerate(show):
        for c, flag in enumerate([False, True]):
            ks = [k for k in m.keys if (per_model[m.key][k]["swap_any"] > 0) == flag]
            vals = [1 - per_model[m.key][k]["oks"] for k in ks]
            mu, lo, hi = boot_mean_ci(vals)
            xx = c + (i - (len(show) - 1) / 2) * w
            if ks:
                ax.bar(xx, 100 * mu, width=w * 0.9, color=m.color)
                ax.errorbar(xx, 100 * mu, yerr=[[100 * (mu - lo)], [100 * (hi - mu)]], fmt="none",
                            ecolor=ps.INK2, elinewidth=0.6)
            rows.append({"model": m.label, "key": m.key, "factor": "lr_swap", "category": "yes" if flag else "no",
                         "n_images": len(ks), "mean_1_minus_oks": mu, "ci_lo": lo, "ci_hi": hi})
    n_sw = [sum(1 for k in show[0].keys if (per_model[show[0].key][k]["swap_any"] > 0) == f) for f in (False, True)]
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"no\nn={n_sw[0]}", f"yes\nn={n_sw[1]}"], fontsize=6.5)
    ax.set_title(f"L/R swap, flagged on {show[0].label}", loc="left", fontsize=7.5)
    ax.grid(axis="x", visible=False)
    for ax in (grid[0][0], grid[1][0]):
        ax.set_ylabel("mean 1 − OKS (%)")
    h, l = axes[0].get_legend_handles_labels()
    leg_ax.axis("off")
    leg_ax.legend(h, l, loc="center", title="bars: mean over images\nwhiskers: 95% bootstrap CI",
                  title_fontsize=6.5)
    fig.tight_layout()
    return ps.save(fig, out), rows


def fig_error_vs_orientation(models, focus, ann, per_model, out):
    show = [m for m in models if (not focus) or m.key in focus][:3]
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.4), gridspec_kw={"width_ratios": [2.2, 1]})
    edges = np.arange(0, 181, 30)
    base = show[0]
    a_abs = np.array([abs(ann[k]["angle_deg"]) if ann[k]["angle_deg"] is not None else np.nan for k in base.keys])
    err0 = np.array([per_model[base.key][k]["err"] for k in base.keys])
    axes[0].scatter(a_abs, err0, s=6, color=ps.CONTEXT_GREY, lw=0, alpha=0.7, label=f"{base.label}: per image")
    rows = []
    for m in show:
        a = np.array([abs(ann[k]["angle_deg"]) if ann[k]["angle_deg"] is not None else np.nan for k in m.keys])
        e = np.array([per_model[m.key][k]["err"] for k in m.keys])
        cx, mu, lo, hi = [], [], [], []
        for b0, b1 in zip(edges[:-1], edges[1:]):
            sel = (a >= b0) & (a < b1 if b1 < 180 else a <= b1)
            if sel.sum() == 0:
                continue
            u, l, h = boot_mean_ci(e[sel])
            cx.append((b0 + b1) / 2)
            mu.append(u)
            lo.append(l)
            hi.append(h)
            rows.append({"model": m.label, "key": m.key, "abs_angle_bin": f"{b0}-{b1}", "n": int(sel.sum()),
                         "mean_err_norm": u, "ci_lo": l, "ci_hi": h})
        axes[0].errorbar(cx, mu, yerr=[np.array(mu) - lo, np.array(hi) - mu], color=m.color, marker=ps.marker(m.slot),
                         ms=4, lw=1.4, elinewidth=0.6, capsize=0, label=f"{m.label}: bin mean (95% CI)")
    axes[0].set_xlabel("|in-plane torso angle| (deg; 0 = head up, 180 = upside down)")
    axes[0].set_ylabel("mean keypoint error / sqrt(area)")
    axes[0].set_xticks(edges)
    axes[0].legend(loc="upper left", fontsize=6)
    w = 0.8 / max(1, len(show))
    use_post = factors_for(ann)[1]
    fkey = "posture" if use_post else "view"
    cats = POSTURES if use_post else ["front", "back", "profile"]
    for i, m in enumerate(show):
        for c, cat in enumerate(cats):
            ks = [k for k in m.keys if ann[k][fkey] == cat]
            u, l, h = boot_mean_ci([per_model[m.key][k]["err"] for k in ks])
            if ks:
                xx = c + (i - (len(show) - 1) / 2) * w
                axes[1].bar(xx, u, width=w * 0.9, color=m.color)
                axes[1].errorbar(xx, u, yerr=[[u - l], [h - u]], fmt="none", ecolor=ps.INK2, elinewidth=0.6)
    ns = [sum(1 for k in base.keys if ann[k][fkey] == c) for c in cats]
    names = cats if use_post else ["front\n(≈supine)", "back\n(≈prone)", "profile\n(≈side)"]
    axes[1].set_xticks(range(len(cats)))
    axes[1].set_xticklabels([f"{c}\nn={n}" for c, n in zip(names, ns)], fontsize=6)
    axes[1].set_title("by posture (GT label)" if use_post else "by view (proxy)", loc="left")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    return ps.save(fig, out), rows


def fig_oks_scatter(ma, mb, ann, out):
    ka = [k for k in ma.keys if k in set(mb.keys)]
    oa = np.array([ma.oks(k) for k in ka])
    ob = np.array([mb.oks(k) for k in ka])
    fig, ax = plt.subplots(figsize=(ps.COL_W, ps.COL_W * 0.95))
    lo = max(0.0, min(oa.min(), ob.min()) - 0.02)
    ax.plot([lo, 1], [lo, 1], color=ps.AXIS, lw=0.8, zorder=1)
    ax.scatter(oa, ob, s=9, color=mb.color, lw=0.4, edgecolor="white", zorder=3)
    d = ob - oa
    win, loss = int((d > 1e-6).sum()), int((d < -1e-6).sum())
    ax.text(0.03, 0.97, f"{mb.label} better: {win} images\n{ma.label} better: {loss} images\n"
                        f"mean ΔOKS = {d.mean():+.4f}", transform=ax.transAxes, va="top", fontsize=6.5, color=ps.INK)
    i = int(np.argmin(d))
    if d[i] < 0:
        ax.annotate(f"largest regression: {ma.rec(ka[i])['file_name'] or ka[i][0]}", (oa[i], ob[i]), fontsize=5.5,
                    color=ps.INK2, xytext=(-4, -10), textcoords="offset points", ha="right")
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo, 1)
    ax.set_xlabel(f"OKS, {ma.label}")
    ax.set_ylabel(f"OKS, {mb.label}")
    fig.tight_layout()
    rows = [{"image_id": k[0], "file_name": ma.rec(k)["file_name"], f"oks_{ma.key}": a, f"oks_{mb.key}": b,
             "delta": b - a, **{f: ann.get(k, {}).get(f) for f in ("inplane", "view", "n_v1", "scale_bin")}}
            for k, a, b in zip(ka, oa, ob)]
    return ps.save(fig, out), rows


def fig_leak_effect(models, focus, leak, lrows, out):
    """Per-image OKS of the Test100 images that are copies of training images (diamonds) vs the rest."""
    fig, ax = plt.subplots(figsize=(ps.COL_W * 1.25, 2.5))
    rng = np.random.default_rng(RNG_SEED)
    for i, m in enumerate(models):
        st = _style(m, focus)
        rest = np.array([m.oks(k) for k in m.keys if k[0] not in leak])
        lk = [(k[0], m.oks(k)) for k in m.keys if k[0] in leak]
        ax.boxplot([rest], positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=ps.GRID, edgecolor=ps.AXIS, lw=0.6), medianprops=dict(color=ps.INK2, lw=0.8),
                   whiskerprops=dict(color=ps.AXIS, lw=0.6), capprops=dict(color=ps.AXIS, lw=0.6))
        ax.scatter(i + rng.uniform(-0.18, 0.18, rest.size), rest, s=4, color=ps.CONTEXT_GREY, lw=0, zorder=2)
        ax.scatter([i + 0.3] * len(lk), [o for _, o in lk], marker="D", s=16, color=st["color"], edgecolor=ps.INK,
                   lw=0.4, zorder=4)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.label for m in models], fontsize=6.5)
    ax.set_xlim(-0.6, len(models) - 0.3)
    ax.set_ylabel("per-image OKS")
    ax.set_title(f"◆ the {len(leak)} Test100 images that are copies of training images; box = the other "
                 f"{len(models[0].keys) - len(leak)}", loc="left", fontsize=6.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return ps.save(fig, out)


def ranking_quality(models, gt_path):
    """AP with the model's own scores vs oracle ranking (score = OKS): how much AP the
    confidence ranking costs. Needs the GT file (re-evaluates with eval_coco, no rescoring)."""
    rows = []
    G = ev.GroundTruth(_p(gt_path)) if gt_path else None
    for m in models:
        for r in range(len(m.runs)):
            keys = m.keys
            sc = np.array([m.rec(k, r)["score"] for k in keys], dtype=float)
            ok = np.array([m.rec(k, r)["oks"] for k in keys], dtype=float)
            rho = _spearman(sc, ok)
            row = {"model": m.label, "key": m.key, "run_id": m.run_ids[r], "spearman_score_oks": rho}
            if G is not None:
                preds = [{"image_id": k[0], "category_id": G.cat_id, "keypoints": np.asarray(m.rec(k, r)["pred_kpts"]).
                          reshape(-1).tolist(), "score": float(m.rec(k, r)["oks"])}
                         for k in keys if m.rec(k, r)["pred_kpts"] is not None]
                for proto in ("paper", "fixed"):
                    o, _ = ev.evaluate(G, preds, rescore=False, primary_protocol=proto, with_per_image=False)
                    row[f"AP_oracle_{proto}"] = o["metrics"]["AP"]
                    met = m.metrics[r]
                    if met:
                        row[f"AP_{proto}"] = (met["metrics"] if proto == "paper" else met["metrics_fixed"])["AP"]
                        row[f"ranking_loss_{proto}"] = row[f"AP_oracle_{proto}"] - row[f"AP_{proto}"]
            rows.append(row)
    return rows


def _spearman(a, b):
    def rank(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[o] = np.arange(len(x))
        # average ties
        for v in np.unique(x):
            idx = np.where(x == v)[0]
            if len(idx) > 1:
                r[idx] = r[idx].mean()
        return r
    if len(a) < 3:
        return None
    ra, rb = rank(np.asarray(a)), rank(np.asarray(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def fig_score_vs_oks(models, focus, out):
    show = [m for m in models if (not focus) or m.key in focus][:3]
    fig, axes = plt.subplots(1, len(show), figsize=(ps.FULL_W * min(1, len(show) / 3 + 0.1), 2.2), sharey=True,
                             squeeze=False)
    for ax, m in zip(axes[0], show):
        sc = np.array([m.rec(k)["score"] for k in m.keys])
        ok = np.array([m.rec(k)["oks"] for k in m.keys])
        ax.scatter(sc, ok, s=7, color=m.color, lw=0.3, edgecolor="white")
        ax.set_title(f"{m.label}  (Spearman ρ = {_spearman(sc, ok):.2f})", loc="left", fontsize=7)
    axes[0][0].set_ylabel("per-image OKS")
    fig.supxlabel("instance score = mean keypoint confidence above 0.2 (the ranking AP uses)", fontsize=7.5)
    fig.tight_layout()
    return ps.save(fig, out)


def swap_rows(models, per_model):
    rows = []
    for m in models:
        cnt = defaultdict(float)
        imgs = []
        for k in m.keys:
            for r in range(len(m.runs)):
                a = tx.annotate(m.rec(k, r))
                for p in a["swapped_pairs"]:
                    cnt[p] += 1.0 / len(m.runs)
            if per_model[m.key][k]["swap_any"] > 0:
                imgs.append(m.rec(k)["file_name"])
        row = {"model": m.label, "key": m.key, "run_ids": m.run_ids,
               "n_images_with_swap": float(sum(per_model[m.key][k]["swap_any"] for k in m.keys)),
               "images": imgs}
        for p in tx.PAIR_NAMES.values():
            row[f"n_{p}"] = cnt.get(p, 0.0)
        rows.append(row)
    return rows


def occlusion_rows(models):
    rows = []
    for m in models:
        for v in (1, 2):
            vals = []
            for k in m.keys:
                e = m.kpt_err(k)
                g = m.rec(k)["gt_kpts"]
                vals += [e[j] for j in range(17) if g[j][2] == v and np.isfinite(e[j])]
            mu, lo, hi = boot_mean_ci(vals)
            rows.append({"model": m.label, "key": m.key, "visibility": "v=1 (occluded)" if v == 1 else "v=2 (visible)",
                         "n_joints": len(vals), "mean_err_norm": mu, "ci_lo": lo, "ci_hi": hi,
                         "pck@0.05": float(np.mean(np.array(vals) < 0.05)) if vals else None,
                         "pck@0.1": float(np.mean(np.array(vals) < 0.1)) if vals else None})
    return rows


def cmd_models(a):
    with open(a.manifest) as f:
        man = json.load(f)
    split = man.get("split", "split")
    models = [Model(s) for s in man["models"]]
    focus = man.get("focus") or [m.key for m in models[:3]]
    out, tabs = a.outdir, a.tables
    os.makedirs(out, exist_ok=True)
    written = []
    p, rows = fig_ap_vs_threshold(models, focus, man.get("published"), os.path.join(out, f"ap_vs_oks_threshold_{split}"))
    written += p
    write_csv(os.path.join(tabs, f"ap_per_threshold_{split}.csv"), rows)
    written += fig_oks_distribution(models, focus, os.path.join(out, f"oks_distribution_{split}"))
    p, rows = fig_per_keypoint(models, focus, os.path.join(out, f"per_keypoint_error_{split}"))
    written += p
    write_csv(os.path.join(tabs, f"per_keypoint_{split}.csv"), rows)
    groups = [("Head (nose, eyes, ears)", [0, 1, 2, 3, 4]), ("Shoulders", [5, 6]), ("Elbows", [7, 8]),
              ("Wrists", [9, 10]), ("Hips", [11, 12]), ("Knees", [13, 14]), ("Ankles", [15, 16]),
              ("All", list(range(17)))]
    grows = []
    for m in models:
        errs = np.stack([m.kpt_err(k) for k in m.keys])
        for gname, idx in groups:
            e = errs[:, idx].ravel()
            e = e[np.isfinite(e)]
            grows.append({"model": m.label, "key": m.key, "run_ids": m.run_ids, "group": gname, "n": int(e.size),
                          "median_err_pct": 100 * float(np.median(e)) if e.size else None,
                          "pck@0.05": 100 * float(np.mean(e < 0.05)) if e.size else None,
                          "pck@0.1": 100 * float(np.mean(e < 0.1)) if e.size else None})
    write_csv(os.path.join(tabs, f"per_keypoint_groups_{split}.csv"), grows)
    posture = load_posture(man.get("posture"))
    ann, per_model, (q1, q2) = taxonomy_table(models, posture)
    if posture:
        cross = defaultdict(lambda: defaultdict(int))
        for k in models[0].keys:
            cross[ann[k]["posture"]][ann[k]["view"]] += 1
        write_csv(os.path.join(tabs, f"view_proxy_vs_posture_{split}.csv"),
                  [{"posture": p, **{f"proxy_{v}": cross[p][v] for v in ("front", "back", "profile", "unknown")}}
                   for p in POSTURES + ["unknown"] if p in cross])
    p, rows = fig_taxonomy(models, focus, ann, per_model, os.path.join(out, f"failure_taxonomy_{split}"))
    written += p
    write_csv(os.path.join(tabs, f"failure_taxonomy_{split}.csv"), rows)
    p, rows = fig_error_vs_orientation(models, focus, ann, per_model, os.path.join(out, f"error_vs_orientation_{split}"))
    written += p
    write_csv(os.path.join(tabs, f"error_vs_orientation_{split}.csv"), rows)
    written += fig_score_vs_oks(models, focus, os.path.join(out, f"score_vs_oks_{split}"))
    write_csv(os.path.join(tabs, f"lr_swaps_{split}.csv"), swap_rows(models, per_model))
    write_csv(os.path.join(tabs, f"occlusion_{split}.csv"), occlusion_rows(models))
    write_csv(os.path.join(tabs, f"ranking_quality_{split}.csv"), ranking_quality(models, man.get("gt")))
    leak = set(int(x) for x in man.get("leak_ids", []))
    if leak:          # memorisation check: images that are copies of training images vs the rest
        lrows = []
        for m in models:
            o_in = [m.oks(k) for k in m.keys if k[0] in leak]
            o_out = [m.oks(k) for k in m.keys if k[0] not in leak]
            mu_in, mu_out = (float(np.mean(o_in)) if o_in else None), (float(np.mean(o_out)) if o_out else None)
            rng = np.random.default_rng(RNG_SEED)
            diffs = [] if not (o_in and o_out) else [
                np.mean(rng.choice(o_in, len(o_in))) - np.mean(rng.choice(o_out, len(o_out))) for _ in range(B_CI)]
            lrows.append({"model": m.label, "key": m.key, "run_ids": m.run_ids, "n_leak": len(o_in),
                          "n_rest": len(o_out), "mean_oks_leak": mu_in, "mean_oks_rest": mu_out,
                          "diff": None if mu_in is None or mu_out is None else mu_in - mu_out,
                          "diff_ci_lo": float(np.percentile(diffs, 2.5)) if diffs else None,
                          "diff_ci_hi": float(np.percentile(diffs, 97.5)) if diffs else None,
                          "leak_ids": sorted(leak), "per_image_oks_leak": [round(x, 4) for x in o_in]})
        write_csv(os.path.join(tabs, f"leak_effect_{split}.csv"), lrows)
        written += fig_leak_effect(models, focus, leak, lrows, os.path.join(out, f"leak_effect_{split}"))
    img_rows = []
    for k in models[0].keys:
        r = dict(image_id=k[0], file_name=models[0].rec(k)["file_name"],
                 **{f: ann[k][f] for f in ("posture", "inplane", "view", "angle_deg", "view_r", "n_v1", "sqrt_area",
                                           "rel_scale", "scale_bin", "occ_bin")})
        for m in models:
            if k in per_model[m.key]:
                r[f"oks_{m.key}"] = per_model[m.key][k]["oks"]
                r[f"swaps_{m.key}"] = per_model[m.key][k]["swapped_pairs"]
                r[f"mirror_gain_{m.key}"] = per_model[m.key][k]["full_swap_gain"]
        img_rows.append(r)
    write_csv(os.path.join(tabs, f"per_image_taxonomy_{split}.csv"), img_rows)
    for pa, pb in man.get("pairs", []):
        ma = next(m for m in models if m.key == pa)
        mb = next(m for m in models if m.key == pb)
        p, rows = fig_oks_scatter(ma, mb, ann, os.path.join(out, f"oks_scatter_{pa}_vs_{pb}_{split}"))
        written += p
        write_csv(os.path.join(tabs, f"oks_pairs_{pa}_vs_{pb}_{split}.csv"), rows)
        # where does the improvement come from? mean ΔOKS by taxonomy category
        drows = []
        for fk, _, cats in factors_for(ann)[0] + ([FACTORS[1]] if factors_for(ann)[1] else []):
            for cat in cats:
                ks = [k for k in ma.keys if k in set(mb.keys) and ann[k][fk] == cat]
                d = [mb.oks(k) - ma.oks(k) for k in ks]
                mu, lo, hi = boot_mean_ci(d)
                drows.append({"a": pa, "b": pb, "factor": fk, "category": cat, "n": len(ks), "mean_delta_oks": mu,
                              "ci_lo": lo, "ci_hi": hi, "share_of_total_gain": float(sum(d)) / max(1e-12, abs(sum(
                                  mb.oks(k) - ma.oks(k) for k in ma.keys if k in set(mb.keys))))})
        write_csv(os.path.join(tabs, f"delta_by_category_{pa}_vs_{pb}_{split}.csv"), drows)
    with open(os.path.join(tabs, f"analysis_manifest_{split}.json"), "w") as f:
        json.dump({"manifest": man, "figures": written, "scale_tertiles_sqrt_area_px": [q1, q2]}, f, indent=2)
    print("\n".join(written))


# ----------------------------------------------------------------------------- splits
def cmd_splits(a):
    sets = []
    for spec in (a.a, a.b):
        name, path = spec.split("=", 1)
        recs, meta = load_per_image(path)
        sets.append((name, recs, load_metrics(path.replace("per_image_", "metrics_"))))
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_W, 2.3), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    t = np.linspace(0.5, 1.0, 201)
    for i, (name, recs, met) in enumerate(sets):
        o = np.array([r["oks"] for r in recs])
        anns = [tx.annotate(r) for r in recs]
        col = SPLIT_COLORS[i]
        axes[0].plot(t, [(o >= x).mean() * 100 for x in t], color=col, lw=1.5, ls="-" if i == 0 else "-",
                     label=f"{name} (n={len(o)})")
        rel = np.array([x["rel_scale"] for x in anns if x["rel_scale"]])
        axes[2].hist(rel, bins=np.linspace(0, 1, 21), histtype="step", color=col, lw=1.3, density=True, label=name)
        row = OrderedDict(split=name, label=a.label, n_images=len(recs), mean_oks=float(o.mean()),
                          median_oks=float(np.median(o)), frac_oks_ge_0_9=float((o >= 0.9).mean()),
                          frac_oks_ge_0_95=float((o >= 0.95).mean()))
        if met:
            row.update(AP_paper=met["metrics"]["AP"], AP_fixed=met["metrics_fixed"]["AP"],
                       AP_80_95=met["metrics"]["AP_80_95"], AP_80_95_fixed=met["metrics_fixed"]["AP_80_95"])
        for f, cats in (("inplane", ["upright", "sideways", "inverted"]), ("view", ["front", "back", "profile"])):
            for c in cats:
                row[f"frac_{f}_{c}"] = float(np.mean([x[f] == c for x in anns]))
        row["frac_any_v1"] = float(np.mean([x["n_v1"] > 0 for x in anns]))
        row["mean_n_v1"] = float(np.mean([x["n_v1"] for x in anns]))
        row["median_sqrt_area_px"] = float(np.median([x["sqrt_area"] for x in anns]))
        row["median_rel_scale"] = float(np.median(rel)) if rel.size else None
        rows.append(row)
    cats = ["side-\nways", "upside\ndown", "back\nview", "profile\nview", "any\nv=1"]
    w = 0.38
    for i, row in enumerate(rows):
        vals = [row["frac_inplane_sideways"], row["frac_inplane_inverted"], row["frac_view_back"],
                row["frac_view_profile"], row["frac_any_v1"]]
        axes[1].bar(np.arange(len(cats)) + (i - 0.5) * w, 100 * np.array(vals), width=w * 0.92,
                    color=SPLIT_COLORS[i], label=row["split"])
    axes[1].set_xticks(range(len(cats)))
    axes[1].set_xticklabels(cats, fontsize=6.5)
    axes[1].set_ylabel("images (%)")
    axes[1].set_title("pose composition (from GT)", loc="left")
    axes[1].grid(axis="x", visible=False)
    axes[0].set_xlabel("OKS threshold t")
    axes[0].set_ylabel("images with OKS ≥ t (%)")
    axes[0].set_title(f"{a.label}: OKS survival", loc="left")
    axes[0].legend(loc="lower left")
    axes[2].set_xlabel("sqrt(box area / image area)")
    axes[2].set_title("relative scale", loc="left")
    axes[2].set_yticks([])
    fig.tight_layout()
    written = ps.save(fig, os.path.join(a.outdir, f"split_difficulty_{a.name}"))
    write_csv(os.path.join(a.tables, f"split_difficulty_{a.name}.csv"), rows)
    print("\n".join(written))


# ----------------------------------------------------------------------------- training curves
def _read_epochs(run_dir):
    """Per-epoch table from upstream epochs.csv or MMPose vis_data/scalars.json (json lines)."""
    p = os.path.join(run_dir, "epochs.csv")
    if os.path.exists(p):
        with open(p) as f:
            rows = list(csv.DictReader(f))
        return [{k: _num(v) for k, v in r.items()} for r in rows], "epochs.csv"
    cands = []
    for root, _, files in os.walk(run_dir):
        for fn in files:
            if fn == "scalars.json":
                cands.append(os.path.join(root, fn))
    if not cands:
        return None, None
    by_epoch = defaultdict(dict)
    train_losses = defaultdict(list)
    for line in open(sorted(cands)[-1]):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        ep = d.get("epoch")
        if "loss" in d and ep is not None:
            train_losses[int(ep)].append(float(d["loss"]))
        for k in ("coco/AP", "coco/AR", "coco/AP .5", "coco/AP .75", "PCK", "AUC"):
            if k in d:
                e = int(d.get("epoch", d.get("step", 0)))
                by_epoch[e][k.replace("coco/", "val_")] = float(d[k])
    rows = []
    for e in sorted(set(train_losses) | set(by_epoch)):
        r = {"epoch": e}
        if train_losses.get(e):
            r["train_loss"] = float(np.mean(train_losses[e]))
        r.update(by_epoch.get(e, {}))
        rows.append(r)
    return rows, os.path.relpath(sorted(cands)[-1], run_dir)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _col(rows, *names):
    for n in names:
        if rows and any(n in r and isinstance(r[n], float) for r in rows):
            return n
    return None


def cmd_curves(a):
    fig, axes = plt.subplots(2, 1, figsize=(ps.COL_W * 1.25, 3.6), sharex=True)
    summary = []
    groups = OrderedDict()
    for spec in a.runs:
        # "label#seed=PATH": split on the LAST '=' so labels may contain '=' (e.g. "λ=5e-4")
        label, path = (spec.rsplit("=", 1) if "=" in spec else (os.path.basename(spec.rstrip("/")), spec))
        groups.setdefault(label.split("#")[0], []).append((label, _p(path)))
    any_val_loss = False
    for gi, (gkey, runs) in enumerate(groups.items()):
        # "label@3#s1": explicit categorical slot, so a model keeps its colour across figures
        m_slot = re.search(r"@(\d+)$", gkey)
        glabel = gkey[:m_slot.start()] if m_slot else gkey
        col = ps.color(int(m_slot.group(1))) if m_slot else ps.SLOTS[gi % len(ps.SLOTS)]
        for ri, (label, path) in enumerate(runs):
            rows, src = _read_epochs(path)
            if not rows:
                print(f"WARNING: no epochs.csv/scalars.json in {path}")
                continue
            ec = _col(rows, "epoch") or "epoch"
            # upstream epochs.csv: train_mse is the pose MSE without the -LAMBDA*BCE term, i.e. the same
            # quantity as val_loss -> the like-for-like over/underfitting comparison
            tl = _col(rows, "train_mse", "train_loss", "loss")
            vset = sorted({str(r.get("val_set")) for r in rows if r.get("val_set") not in (None, "")})
            if any(("test" in v.lower() or v in ("validate_infant", "validate100")) for v in vset):
                print(f"WARNING {path}: per-epoch AP computed on {vset} (Test100?) -- NOT a selection signal")
            vl = _col(rows, "val400_loss", "val_loss", "valid_loss")
            ap = _col(rows, "val400_AP", "val_AP", "AP")
            ar = _col(rows, "val400_AR", "val_AR", "AR")
            ep = np.array([r.get(ec) for r in rows], dtype=float)
            first = ri == 0
            if tl:
                y = np.array([r.get(tl, np.nan) for r in rows], dtype=float)
                axes[0].plot(ep, y, color=col, lw=1.2, label=f"{glabel}: train" if first else None)
            if vl:
                any_val_loss = True
                y = np.array([r.get(vl, np.nan) for r in rows], dtype=float)
                axes[0].plot(ep, y, color=col, lw=1.2, alpha=0.55, marker=".", ms=3,
                             label=f"{glabel}: Val400" if first else None)
            best = {}
            if ap:
                y = np.array([r.get(ap, np.nan) for r in rows], dtype=float)
                scale = 100 if np.nanmax(y) <= 1.0 else 1
                axes[1].plot(ep, scale * y, color=col, lw=1.3, label=f"{glabel}: AP" if first else None)
                flags = [r.get("is_best") for r in rows]
                if any(isinstance(f, float) and f == 1.0 for f in flags):
                    b = max(i for i, f in enumerate(flags) if isinstance(f, float) and f == 1.0)   # upstream rule
                else:
                    b = int(np.nanargmax(y))
                axes[1].plot(ep[b], scale * y[b], marker="o", ms=4.5, mfc="white", mec=col, zorder=5)
                best = {"best_epoch": ep[b], "best_val_AP": float(y[b]), "final_epoch": ep[-1],
                        "final_val_AP": float(y[~np.isnan(y)][-1]), "val_set": ";".join(vset),
                        "train_loss_col": tl, "gap_final_val_minus_train": (
                            float(rows[-1][vl] - rows[-1][tl]) if (vl and tl and isinstance(rows[-1].get(vl), float)
                                                                   and isinstance(rows[-1].get(tl), float)) else None)}
            if ar:
                y = np.array([r.get(ar, np.nan) for r in rows], dtype=float)
                scale = 100 if np.nanmax(y) <= 1.0 else 1
                axes[1].plot(ep, scale * y, color=col, lw=0.9, alpha=0.55, label=f"{glabel}: AR" if first else None)
            summary.append({"group": glabel, "run": label, "run_id": ev.infer_run_id(path + "/"), "source": src,
                            "n_epochs": len(rows), **best,
                            "train_loss_first": rows[0].get(tl) if tl else None,
                            "train_loss_last": rows[-1].get(tl) if tl else None,
                            "val_loss_min_epoch": (ep[int(np.nanargmin([r.get(vl, np.nan) for r in rows]))]
                                                   if vl else None)})
    axes[0].set_ylabel("heatmap MSE")
    axes[0].set_title("train vs Val400 pose loss (same MSE)" if any_val_loss else
                      "training loss (no validation loss logged)", loc="left")
    axes[0].legend(fontsize=6)
    from matplotlib.ticker import MaxNLocator
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_ylabel("Val400 (%)")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Val400 AP / AR per epoch (○ = best, the selection rule)", loc="left")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    written = ps.save(fig, os.path.join(a.outdir, f"training_curves_{a.name}"))
    write_csv(os.path.join(a.tables, f"training_curves_{a.name}.csv"), summary)
    print("\n".join(written))


# ----------------------------------------------------------------------------- GT-box source effect
def cmd_gtbox(a):
    """2x2: predictions made with v100 vs v500 GT boxes, scored against v100 vs v500 GT, on the 100 shared images."""
    v100 = json.load(open(_p(a.gt_v100)))
    v500 = json.load(open(_p(a.gt_v500)))
    ids = {im["id"] for im in v100["images"]}
    sub500 = dict(v500, images=[im for im in v500["images"] if im["id"] in ids],
                  annotations=[x for x in v500["annotations"] if x["image_id"] in ids])
    rows = []
    for pname, ppath in (("v100 boxes", a.pred_v100box), ("v500 boxes", a.pred_v500box)):
        if not ppath:
            continue
        for gname, g in (("v100 GT", v100), ("v500 GT", sub500)):
            o, _ = ev.evaluate(json.loads(json.dumps(g)), _p(ppath), with_per_image=False)
            rows.append({"pred_boxes": pname, "gt": gname, "pred": ppath, "run_id": o["run_id"],
                         "AP_paper": o["metrics"]["AP"], "AP_fixed": o["metrics_fixed"]["AP"],
                         "AP_80_95_fixed": o["metrics_fixed"]["AP_80_95"], "mean_oks": o["oks"]["mean"],
                         "n_dropped": o["pred"]["n_dropped_not_in_gt"]})
    a1 = {x["image_id"]: x for x in v100["annotations"]}
    a5 = {x["image_id"]: x for x in sub500["annotations"]}
    ratio = [(a1[i]["bbox"][2] * a1[i]["bbox"][3]) / (a5[i]["bbox"][2] * a5[i]["bbox"][3]) for i in a1 if i in a5]
    rows.append({"pred_boxes": "box-area ratio v100/v500", "gt": "stats", "pred": "",
                 "AP_paper": float(np.median(ratio)), "AP_fixed": float(np.min(ratio)), "AP_80_95_fixed": float(np.max(ratio)),
                 "mean_oks": None, "n_dropped": len(ratio), "run_id": "median/min/max in AP_paper/AP_fixed/AP_80_95_fixed"})
    write_csv(os.path.join(a.tables, "gtbox_v100_vs_v500.csv"), rows)
    for r in rows:
        print(f"{r['pred_boxes']:11s} x {r['gt']:8s}: AP paper {100 * r['AP_paper']:.2f}  fixed {100 * r['AP_fixed']:.2f}"
              f"  mean OKS {r['mean_oks']:.4f}")


# ----------------------------------------------------------------------------- id-0 rank sensitivity (D13)
def id0_rank_curve(G, pred):
    """Paper/fixed AP of one prediction set as test0's detection is moved to every score rank.
    All other scores are untouched. Uses the shared rescoring for the actual scores."""
    preds, _, _ = ev.load_predictions(_p(pred), G)
    inst = ev.postprocess(preds)                     # rescored, OKS-NMS
    zero = [a for a in G.dataset["annotations"] if a["id"] == 0]
    if not zero:
        raise ValueError("GT has no annotation id 0")
    img0 = zero[0]["image_id"]
    mine = [i for i, p in enumerate(inst) if p["image_id"] == img0]
    if len(mine) != 1:
        raise ValueError(f"expected exactly one detection on image {img0}, got {len(mine)}")
    k0 = mine[0]
    others = np.sort(np.array([p["score"] for i, p in enumerate(inst) if i != k0]))[::-1]
    N = len(inst)
    actual = 1 + int(np.sum(others > inst[k0]["score"]))
    rows = []
    for r in range(1, N + 1):
        if r == 1:
            s0 = others[0] + 1e-3
        elif r == N:
            s0 = others[-1] - 1e-3
        else:
            s0 = 0.5 * (others[r - 2] + others[r - 1])
        res = [dict(p) for p in inst]
        res[k0]["score"] = float(s0)
        row = {"rank": r, "actual": r == actual}
        for proto in ("paper", "fixed"):
            _, m, _, _ = ev.run_cocoeval(G.coco(proto), ev._to_coco_results(res))
            row[f"AP_{proto}"], row[f"AP50_{proto}"], row[f"AR_{proto}"] = m["AP"], m["AP50"], m["AR"]
        rows.append(row)
    return rows, actual, N


def cmd_id0rank(a):
    G = ev.GroundTruth(_p(a.gt))
    fig, ax = plt.subplots(figsize=(ps.COL_W * 1.15, 2.9))
    table = []
    for i, spec in enumerate(a.model):
        label, rest = spec.split("=", 1)
        path, slot = (rest.rsplit(":", 1) if re.search(r":\d+$", rest) else (rest, i + 1))
        rows, actual, N = id0_rank_curve(G, path)
        col = ps.color(int(slot))
        x = np.array([r["rank"] for r in rows])
        yp = 100 * np.array([r["AP_paper"] for r in rows])
        yf = 100 * np.array([r["AP_fixed"] for r in rows])
        ax.plot(x, yp, color=col, lw=1.4, label=f"{label}: paper protocol")
        ax.plot(x, yf, color=col, lw=0.9, alpha=0.45, label=f"{label}: fixed protocol")
        ax.plot([actual], [yp[actual - 1]], marker="o", ms=5, mfc="white", mec=col, zorder=5)
        ax.annotate(f"actual rank {actual}: {yp[actual - 1]:.2f}", (actual, yp[actual - 1]), xytext=(4, -10),
                    textcoords="offset points", fontsize=6, color=ps.INK2)
        for r in rows:
            table.append({"model": label, "pred": path, "run_id": ev.infer_run_id(path), **r})
        table.append({"model": label, "pred": path, "run_id": ev.infer_run_id(path), "rank": "summary",
                      "actual": actual, "AP_paper": float(yp[actual - 1] / 100), "AP_fixed": float(yf[actual - 1] / 100),
                      "AP50_paper": float((yp.max() - yp.min()) / 100), "AR_paper": None,
                      "AP50_fixed": None, "AR_fixed": None})
        print(f"{label}: actual rank {actual}/{N}; paper AP {yp[actual - 1]:.2f} (range {yp.min():.2f}-{yp.max():.2f}, "
              f"swing {yp.max() - yp.min():.2f}); fixed AP {yf.min():.2f}-{yf.max():.2f}")
    ax.set_xlabel("score rank of test0's detection (1 = most confident)")
    ax.set_ylabel("Test100 AP (%)")
    ax.set_title("same predictions; only test0's confidence rank moves", loc="left", fontsize=7.5)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06 + 0.045 * ((len(l) + 1) // 2), 1, 1))
    written = ps.save(fig, os.path.join(a.outdir, f"id0_rank_sensitivity_{a.name}"))
    for r in table:
        if r["rank"] == "summary":
            r["note"] = "AP50_paper column holds the paper-AP swing (max-min over ranks) in this summary row"
    write_csv(os.path.join(a.tables, f"id0_rank_sensitivity_{a.name}.csv"), table)
    print("\n".join(written))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("id0rank")
    s.add_argument("--gt", required=True)
    s.add_argument("--model", action="append", required=True, help="label=preds.json[:slot] (repeatable)")
    s.add_argument("--name", required=True)
    s.add_argument("--outdir", required=True)
    s.add_argument("--tables", required=True)
    s = sub.add_parser("models")
    s.add_argument("--manifest", required=True)
    s.add_argument("--outdir", required=True)
    s.add_argument("--tables", required=True)
    s = sub.add_parser("splits")
    s.add_argument("--label", required=True)
    s.add_argument("--name", default="released")
    s.add_argument("--a", required=True, help="name=per_image.json")
    s.add_argument("--b", required=True, help="name=per_image.json")
    s.add_argument("--outdir", required=True)
    s.add_argument("--tables", required=True)
    s = sub.add_parser("curves")
    s.add_argument("--runs", nargs="+", required=True,
                   help="[label[@slot]#seed=]RUN_DIR (same label -> same colour; @slot pins the model's colour)")
    s.add_argument("--name", required=True)
    s.add_argument("--outdir", required=True)
    s.add_argument("--tables", required=True)
    s = sub.add_parser("gtbox")
    s.add_argument("--pred-v100box", required=True)
    s.add_argument("--pred-v500box", default=None)
    s.add_argument("--gt-v100", required=True)
    s.add_argument("--gt-v500", required=True)
    s.add_argument("--tables", required=True)
    a = ap.parse_args(argv)
    {"models": cmd_models, "splits": cmd_splits, "curves": cmd_curves, "gtbox": cmd_gtbox,
     "id0rank": cmd_id0rank}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
