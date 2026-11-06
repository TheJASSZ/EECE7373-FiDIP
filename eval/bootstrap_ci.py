#!/usr/bin/env python
"""bootstrap_ci.py -- paired image-level bootstrap for keypoint AP differences.

Modes (all share one GT and one matching protocol; default 'paper', as the headline tables):

  paired     ΔAP between two prediction files on the same images:
             python eval/bootstrap_ci.py paired --gt GT --a A.json --b B.json --out ci.json
  multiseed  A and B are lists of seed runs: mean ± std across seeds, plus the bootstrap over
             images of the seed-averaged AP (same resample for every file):
             python eval/bootstrap_ci.py multiseed --gt GT --a A_s1 A_s2 A_s3 --b B_s1 B_s2 B_s3 --out ci.json
  contrasts  a declared list of contrasts (JSON spec) -> per-contrast CI + Holm-adjusted p-values:
             python eval/bootstrap_ci.py contrasts --gt GT --spec contrasts.json --out ci.json
  single     CI of one model's AP:  python eval/bootstrap_ci.py single --gt GT --a A.json --out ci.json

Statistic: COCO keypoint AP (also AP50, AP75, AP_80_95, AR) of the resampled image set.

Fast path (exact):
- Per-image matching is taken verbatim from COCOeval.evaluate()'s `evalImgs` (area 'all', maxDets 20).
- Only COCOeval.accumulate() is re-implemented (vectorised), so the AP of any multiset of images is
  exactly what COCOeval would report for that multiset.
- Before any resampling, the full-set fast AP is asserted equal to COCOeval's stats; this
  self-check is recorded in the output.

Inference:
- 95% percentile CI of Δ = AP_A - AP_B over B resamples (default 10000, seed 0).
- One-sided p for H1: A > B, as (1 + #{Δ* <= 0}) / (B + 1).
- Holm step-down adjustment across declared contrasts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_coco as ev  # noqa: E402

STAT_NAMES = ("AP", "AP50", "AP75", "AP_80_95", "AR", "AR50", "AR75")


# ----------------------------------------------------------------------------- per-image cache
class PerImageEval:
    """COCOeval's per-image match results for one prediction file, plus a vectorised accumulate."""

    def __init__(self, G, pred, protocol="paper", rescore=True, kpt_thr=ev.DEFAULT_KPT_THR,
                 nms_thr=ev.DEFAULT_NMS_THR, nms="oks", label=None):
        self.label = label or (pred if isinstance(pred, str) else "preds")
        self.path = pred if isinstance(pred, str) else None
        preds, self.pred_info, _ = ev.load_predictions(pred, G)
        inst = ev.postprocess(preds, rescore, kpt_thr, nms_thr, nms)
        E, metrics, _, _ = ev.run_cocoeval(G.coco(protocol), ev._to_coco_results(inst))
        if E is None:
            raise ValueError(f"{self.label}: no predictions on this GT")
        self.metrics = metrics
        p = E.params
        self.img_ids = list(p.imgIds)
        self.iou_thrs = np.asarray(p.iouThrs, dtype=float)
        self.rec_thrs = np.asarray(p.recThrs, dtype=float)
        self.t50 = int(np.where(np.isclose(self.iou_thrs, 0.5))[0][0])
        self.t75 = int(np.where(np.isclose(self.iou_thrs, 0.75))[0][0])
        self.strict = np.array([any(abs(t - s) < 1e-9 for s in ev.STRICT_THRS) for t in self.iou_thrs])
        a = p.areaRngLbl.index("all")
        max_det = p.maxDets[-1]
        n_img, n_area = len(p.imgIds), len(p.areaRng)
        T = len(self.iou_thrs)
        n_det, npig, scores, tp, fp = [], [], [], [], []
        for i in range(n_img):
            e = E.evalImgs[a * n_img + i]          # single category: k = 0
            if e is None:
                n_det.append(0)
                npig.append(0)
                continue
            s = np.asarray(e["dtScores"][:max_det], dtype=float)
            dtm = np.asarray(e["dtMatches"])[:, :max_det]
            dtig = np.asarray(e["dtIgnore"])[:, :max_det].astype(bool)
            # pycocotools: tps = logical_and(dtm, ~dtIg) -> a match to GT id 0 is NOT a TP ('paper')
            matched = dtm != 0
            n_det.append(len(s))
            npig.append(int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)))
            scores.append(s)
            tp.append(np.logical_and(matched, ~dtig).reshape(T, -1))
            fp.append(np.logical_and(~matched, ~dtig).reshape(T, -1))
        self.n_det = np.asarray(n_det, dtype=np.int64)
        self.npig = np.asarray(npig, dtype=np.int64)
        self.start = np.concatenate([[0], np.cumsum(self.n_det)[:-1]]).astype(np.int64)
        self.scores = np.concatenate(scores) if scores else np.zeros(0)
        self.tp = np.concatenate(tp, axis=1) if tp else np.zeros((T, 0), bool)
        self.fp = np.concatenate(fp, axis=1) if fp else np.zeros((T, 0), bool)
        self.coco_stats = OrderedDict((k, metrics[k]) for k in STAT_NAMES)
        self.self_check = self._check()
        # per-image best OKS (protocol independent) for consistency statistics
        recs = ev.per_image_records(G, E, inst)
        oks = {r["image_id"]: r["oks"] for r in recs}
        self.oks = np.array([oks.get(i, 0.0) for i in self.img_ids])

    def _rows(self, sample):
        counts = self.n_det[sample]
        total = int(counts.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        offs = np.repeat(self.start[sample] - (np.cumsum(counts) - counts), counts)
        return offs + np.arange(total)

    def stats(self, sample=None):
        """COCO stats of the multiset of images `sample` (indices into img_ids; None = all, in order)."""
        if sample is None:
            sample = np.arange(len(self.img_ids))
        sample = np.asarray(sample, dtype=np.int64)
        npig = int(self.npig[sample].sum())
        T, R = len(self.iou_thrs), len(self.rec_thrs)
        if npig == 0:
            return OrderedDict((k, None) for k in STAT_NAMES)
        rows = self._rows(sample)
        order = np.argsort(-self.scores[rows], kind="mergesort")
        rows = rows[order]
        tp_sum = np.cumsum(self.tp[:, rows], axis=1).astype(dtype=float)
        fp_sum = np.cumsum(self.fp[:, rows], axis=1).astype(dtype=float)
        nd = rows.size
        precision = np.zeros((T, R))
        recall = np.zeros(T)
        for t in range(T):
            tp, fp = tp_sum[t], fp_sum[t]
            if nd == 0:
                continue
            rc = tp / npig
            pr = tp / (fp + tp + np.spacing(1))
            recall[t] = rc[-1]
            pr = np.maximum.accumulate(pr[::-1])[::-1]       # == COCOeval's right-to-left envelope loop
            inds = np.searchsorted(rc, self.rec_thrs, side="left")
            ok = inds < nd
            precision[t, ok] = pr[inds[ok]]
        return OrderedDict(
            AP=float(np.mean(precision)), AP50=float(np.mean(precision[self.t50])),
            AP75=float(np.mean(precision[self.t75])), AP_80_95=float(np.mean(precision[self.strict])),
            AR=float(np.mean(recall)), AR50=float(recall[self.t50]), AR75=float(recall[self.t75]))

    def _check(self):
        fast = self.stats()
        diffs = {k: abs(fast[k] - self.coco_stats[k]) for k in STAT_NAMES
                 if fast[k] is not None and self.coco_stats[k] is not None}
        worst = max(diffs.values()) if diffs else 0.0
        if worst > 1e-12:
            raise AssertionError(f"fast path != COCOeval for {self.label}: {diffs}")
        return {"max_abs_diff_vs_cocoeval": worst, "passed": True}


# ----------------------------------------------------------------------------- statistics
def _resamples(n, B, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))


def _ci(x, level=0.95):
    lo, hi = np.percentile(x, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return [float(lo), float(hi)]


def _boot_matrix(models, idx):
    """(n_models, B, n_stats) array of resampled stats."""
    out = np.zeros((len(models), idx.shape[0], len(STAT_NAMES)))
    for m, M in enumerate(models):
        for b in range(idx.shape[0]):
            s = M.stats(idx[b])
            out[m, b] = [np.nan if s[k] is None else s[k] for k in STAT_NAMES]
    return out


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def _consistency(oks_a, oks_b, tol=1e-6):
    d = np.asarray(oks_a) - np.asarray(oks_b)
    win, loss = int(np.sum(d > tol)), int(np.sum(d < -tol))
    tie = int(d.size - win - loss)
    out = OrderedDict(n_images=int(d.size), wins=win, ties=tie, losses=loss,
                      mean_delta_oks=float(d.mean()), median_delta_oks=float(np.median(d)))
    n = win + loss
    if n:   # exact two-sided sign test on non-tied images
        k = min(win, loss)
        p = sum(math.comb(n, j) for j in range(0, k + 1)) / 2 ** n
        out["sign_test_p_two_sided"] = float(min(1.0, 2 * p))
    try:
        from scipy.stats import wilcoxon
        if np.any(np.abs(d) > tol):
            out["wilcoxon_p_two_sided"] = float(wilcoxon(d[np.abs(d) > tol]).pvalue)
    except Exception:  # scipy optional
        pass
    return out


def _arm_summary(models, boot_arm_mean, label):
    per_seed = [OrderedDict((k, M.coco_stats[k]) for k in STAT_NAMES) for M in models]
    ap = np.array([s["AP"] for s in per_seed])
    out = OrderedDict(label=label, n_runs=len(models), files=[M.path for M in models],
                      run_ids=[ev.infer_run_id(M.path) for M in models])
    out["mean"] = OrderedDict((k, float(np.mean([s[k] for s in per_seed]))) for k in STAT_NAMES)
    out["std_across_runs"] = OrderedDict(
        (k, float(np.std([s[k] for s in per_seed], ddof=1)) if len(models) > 1 else None) for k in STAT_NAMES)
    out["per_run_AP"] = ap.tolist()
    out["boot_ci95"] = OrderedDict((k, _ci(boot_arm_mean[:, j])) for j, k in enumerate(STAT_NAMES))
    out["self_check"] = [M.self_check for M in models]
    return out


def compare(G, a_files, b_files, B=10000, seed=0, protocol="paper", label_a="A", label_b="B",
            metric="AP", hierarchical=False, both_protocols=True):
    """Paired / multi-seed comparison under `protocol` (headline). With both_protocols (D13), the other
    protocol is computed on the SAME resamples and attached as `sensitivity`. An improvement is robust
    only if its 95% CI excludes 0, with the same sign, under both protocols."""
    out = _compare_one(G, a_files, b_files, B, seed, protocol, label_a, label_b, metric, hierarchical)
    if both_protocols:
        other = "fixed" if protocol == "paper" else "paper"
        o2 = _compare_one(G, a_files, b_files, B, seed, other, label_a, label_b, metric, hierarchical)
        out["sensitivity"] = OrderedDict(
            protocol=other, delta=o2["delta"], p_one_sided_a_gt_b=o2["p_one_sided_a_gt_b"],
            p_one_sided_b_gt_a=o2["p_one_sided_b_gt_a"], p_two_sided=o2["p_two_sided"],
            a_mean=o2["a"]["mean"], b_mean=o2["b"]["mean"], self_check=[x["self_check"] for x in (o2["a"], o2["b"])])
        lo1, hi1 = out["delta"][metric]["ci95"]
        lo2, hi2 = o2["delta"][metric]["ci95"]
        out["a_gt_b_robust_both_protocols"] = bool(lo1 > 0 and lo2 > 0)
        out["ci_excludes_zero_both_protocols"] = bool((lo1 > 0 and lo2 > 0) or (hi1 < 0 and hi2 < 0))
    return out


def _compare_one(G, a_files, b_files, B=10000, seed=0, protocol="paper", label_a="A", label_b="B",
                 metric="AP", hierarchical=False):
    """Paired (1 file per arm) or multi-seed (lists) comparison on the same resampled images."""
    A = [PerImageEval(G, f, protocol, label=f) for f in a_files]
    Bm = [PerImageEval(G, f, protocol, label=f) for f in b_files]
    ids = A[0].img_ids
    for M in A + Bm:
        if M.img_ids != ids:
            raise ValueError("all prediction files must be evaluated on the same GT images")
    idx = _resamples(len(ids), B, seed)
    boot_a = _boot_matrix(A, idx)          # (ka, B, S)
    boot_b = _boot_matrix(Bm, idx)
    mean_a, mean_b = boot_a.mean(0), boot_b.mean(0)       # seed-averaged per resample
    delta = mean_a - mean_b
    j = STAT_NAMES.index(metric)
    obs_a = np.array([[M.coco_stats[k] for k in STAT_NAMES] for M in A]).mean(0)
    obs_b = np.array([[M.coco_stats[k] for k in STAT_NAMES] for M in Bm]).mean(0)
    obs = obs_a - obs_b
    out = OrderedDict()
    out["mode"] = "paired" if len(A) == len(Bm) == 1 else "multiseed"
    out["protocol"] = protocol
    out["gt"] = G.path
    out["n_images"] = len(ids)
    out["B"] = int(B)
    out["seed"] = int(seed)
    out["metric"] = metric
    out["a"] = _arm_summary(A, mean_a, label_a)
    out["b"] = _arm_summary(Bm, mean_b, label_b)
    d = OrderedDict()
    for jj, k in enumerate(STAT_NAMES):
        d[k] = OrderedDict(observed=float(obs[jj]), boot_mean=float(np.mean(delta[:, jj])),
                           boot_se=float(np.std(delta[:, jj], ddof=1)), ci95=_ci(delta[:, jj]),
                           frac_boot_a_gt_b=float(np.mean(delta[:, jj] > 0)))
    out["delta"] = d
    dm = delta[:, j]
    out["p_one_sided_a_gt_b"] = float((1 + np.sum(dm <= 0)) / (B + 1))
    out["p_one_sided_b_gt_a"] = float((1 + np.sum(dm >= 0)) / (B + 1))
    out["p_two_sided"] = float(min(1.0, 2 * min(out["p_one_sided_a_gt_b"], out["p_one_sided_b_gt_a"])))
    if len(A) > 1 or len(Bm) > 1:
        pa = [M.coco_stats[metric] for M in A]
        pb = [M.coco_stats[metric] for M in Bm]
        out["seed_pairs_a_gt_b"] = float(np.mean([x > y for x in pa for y in pb]))
        out["all_a_runs_gt_all_b_runs"] = bool(min(pa) > max(pb))
        if hierarchical:
            rng = np.random.default_rng(seed + 1)
            sa = rng.integers(0, len(A), size=(B, len(A)))
            sb = rng.integers(0, len(Bm), size=(B, len(Bm)))
            h = np.array([boot_a[sa[b], b, j].mean() - boot_b[sb[b], b, j].mean() for b in range(B)])
            out["hierarchical"] = OrderedDict(ci95=_ci(h), p_one_sided_a_gt_b=float((1 + np.sum(h <= 0)) / (B + 1)),
                                              note="resamples seeds within each arm and images jointly")
    oks_a = np.mean([M.oks for M in A], axis=0)
    oks_b = np.mean([M.oks for M in Bm], axis=0)
    out["per_image_consistency"] = _consistency(oks_a, oks_b)
    return out


def single(G, a_file, B=10000, seed=0, protocol="paper"):
    M = PerImageEval(G, a_file, protocol, label=a_file)
    idx = _resamples(len(M.img_ids), B, seed)
    boot = _boot_matrix([M], idx)[0]
    out = OrderedDict(mode="single", protocol=protocol, gt=G.path, n_images=len(M.img_ids), B=int(B),
                      seed=int(seed), file=a_file, run_id=ev.infer_run_id(a_file))
    out["observed"] = M.coco_stats
    out["boot_ci95"] = OrderedDict((k, _ci(boot[:, j])) for j, k in enumerate(STAT_NAMES))
    out["boot_se"] = OrderedDict((k, float(np.std(boot[:, j], ddof=1))) for j, k in enumerate(STAT_NAMES))
    out["self_check"] = M.self_check
    return out


def run_contrasts(G, spec, B=None, seed=None, protocol=None):
    B = int(B or spec.get("B", 10000))
    seed = int(spec.get("seed", 0) if seed is None else seed)
    protocol = protocol or spec.get("protocol", "paper")
    metric = spec.get("metric", "AP")
    res = []
    for c in spec["contrasts"]:
        a = [os.path.expandvars(x) for x in (c["a"] if isinstance(c["a"], list) else [c["a"]])]
        b = [os.path.expandvars(x) for x in (c["b"] if isinstance(c["b"], list) else [c["b"]])]
        r = compare(G, a, b, B=B, seed=seed, protocol=protocol, label_a=c.get("label_a", "A"),
                    label_b=c.get("label_b", "B"), metric=metric, hierarchical=c.get("hierarchical", False))
        r["name"] = c["name"]
        r["hypothesis"] = c.get("hypothesis")
        res.append(r)
    adj = holm([r["p_one_sided_a_gt_b"] for r in res])
    alpha = float(spec.get("alpha", 0.05))
    for r, pa in zip(res, adj):
        r["p_holm"] = pa
        r["reject_h0_holm"] = bool(pa < alpha)
    if all("sensitivity" in r for r in res):
        adj2 = holm([r["sensitivity"]["p_one_sided_a_gt_b"] for r in res])
        for r, pa in zip(res, adj2):
            r["sensitivity"]["p_holm"] = pa
            r["sensitivity"]["reject_h0_holm"] = bool(pa < alpha)
            r["robust_claim"] = bool(r["reject_h0_holm"] and r["sensitivity"]["reject_h0_holm"]
                                     and r["a_gt_b_robust_both_protocols"])
    return OrderedDict(mode="contrasts", protocol=protocol, metric=metric, B=B, seed=seed, alpha=alpha,
                       n_contrasts=len(res), correction="Holm step-down over the declared contrasts",
                       contrasts=res)


def _fmt_delta(r, metric="AP"):
    d = r["delta"][metric]
    txt = (f"Δ{metric} = {100 * d['observed']:+.2f} [{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}]"
           f"  p(A>B)={r['p_one_sided_a_gt_b']:.4f}")
    if "sensitivity" in r:
        e = r["sensitivity"]["delta"][metric]
        txt += (f"  | {r['sensitivity']['protocol']}: {100 * e['observed']:+.2f} [{100 * e['ci95'][0]:+.2f}, "
                f"{100 * e['ci95'][1]:+.2f}]  robust(both)={r['a_gt_b_robust_both_protocols']}")
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("paired", "multiseed", "contrasts", "single"):
        s = sub.add_parser(name)
        s.add_argument("--gt", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--B", type=int, default=None, help="bootstrap resamples (default 10000)")
        s.add_argument("--seed", type=int, default=None, help="RNG seed (default 0)")
        s.add_argument("--protocol", choices=list(ev.PROTOCOLS), default=None, help="default 'paper'")
        s.add_argument("--area", choices=["gt", "bbox"], default="gt")
        if name == "contrasts":
            s.add_argument("--spec", required=True, help="JSON: {contrasts: [{name, a, b, label_a, label_b}]}")
        else:
            s.add_argument("--a", nargs="+", required=True)
        if name in ("paired", "multiseed"):
            s.add_argument("--b", nargs="+", required=True)
            s.add_argument("--label-a", default="A")
            s.add_argument("--label-b", default="B")
            s.add_argument("--metric", choices=list(STAT_NAMES), default="AP")
            s.add_argument("--hierarchical", action="store_true", help="multiseed: also resample seeds")
    a = ap.parse_args(argv)
    G = ev.GroundTruth(a.gt, area_rule=a.area)
    B = a.B or 10000
    seed = 0 if a.seed is None else a.seed
    protocol = a.protocol or "paper"
    if a.mode in ("paired", "multiseed"):
        if a.mode == "paired" and (len(a.a) != 1 or len(a.b) != 1):
            ap.error("paired takes exactly one file per arm; use multiseed")
        out = compare(G, a.a, a.b, B=B, seed=seed, protocol=protocol, label_a=a.label_a,
                      label_b=a.label_b, metric=a.metric, hierarchical=a.hierarchical)
        print(f"[bootstrap] {a.label_a} vs {a.label_b} ({out['mode']}, {protocol}, n={out['n_images']}, B={B}): "
              + _fmt_delta(out, a.metric))
    elif a.mode == "single":
        out = single(G, a.a[0], B=B, seed=seed, protocol=protocol)
        ci = out["boot_ci95"]["AP"]
        print(f"[bootstrap] AP = {100 * out['observed']['AP']:.2f} [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]")
    else:
        with open(a.spec) as f:
            spec = json.load(f)
        out = run_contrasts(G, spec, B=a.B, seed=a.seed, protocol=a.protocol)
        for r in out["contrasts"]:
            print(f"[bootstrap] {r['name']}: {_fmt_delta(r, out['metric'])}  p_holm={r['p_holm']:.4f}")
    out["settings"] = OrderedDict(evaluator=f"eval/eval_coco.py v{ev.EVALUATOR_VERSION}", area_rule=a.area,
                                  ci="percentile", **ev._versions())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
