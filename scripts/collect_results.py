#!/usr/bin/env python
"""collect_results.py -- build every results table + the LaTeX numbers for the report.

Never hand-type a number in the report. `\\input{generated/results_tables.tex}` and use its macros
and tables. Every value in the CSVs carries the RUN_IDs it came from.

Inputs:
- results/registry.csv: one row per run.
- results/PREREGISTRATION.md: Test100 is reported only for registered run IDs, plus the
  released checkpoints and official-recipe retrains (roles 'released'/'retrain' in configs.json).
- results/configs.json: optional. Groups runs into configurations and gives labels, roles,
  "what changed", the hypothesis and the macro name.
- results/analysis_eval/<RUN_ID>/metrics_<split>.json: eval_coco.py outputs pulled from
  $R/analysis/eval by scripts/pull_analysis.sh. Falls back to results/cluster_runs/<RUN_ID>/metrics_<split>.json.
- results/tables/bootstrap_*.json: bootstrap_ci.py outputs.

Outputs:
- results/tables/{main_comparison,reproduction,attempts,seeds,ablations,bootstrap_summary,per_run}.{csv,json}
- report/generated/results_tables.tex
- report/generated/MACROS.md

Split rules:
- Test100 is the only clean test set.
- Val400 is the selection set.
- validate500 contains Val400 and is therefore never clean; it is flagged and never reported as a test set.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import OrderedDict, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ("test100", "val400", "validate500")
CLEAN = {"test100": True, "val400": False, "validate500": False}
SPLIT_ROLE = {"test100": "clean test (reporting only)", "val400": "selection",
              "validate500": "NOT clean (contains Val400)"}
METRICS = ("AP", "AP50", "AP75", "AR", "AR50", "AR75", "AP_80_95")
# Published numbers: upstream README @ a76b4f4, "Generality of our FiDIP method ... SyRIP Test100".
PUBLISHED = OrderedDict([
    ("paper_fidip_darkpose", {"label": "DarkPose + FiDIP (paper)", "AP": 93.6, "AP50": 98.5, "AP75": 98.5, "AR": 94.6,
                              "AR50": 99.0, "AR75": 99.0, "macro": "PaperFiDIP"}),
    ("paper_finetune_darkpose", {"label": "DarkPose + Finetune (paper)", "AP": 92.7, "AP50": 98.5, "AP75": 98.5,
                                 "AR": 93.9, "AR50": 99.0, "AR75": 99.0, "macro": "PaperFinetune"}),
    ("paper_darkpose", {"label": "DarkPose, no fine-tuning (paper)", "AP": 88.5, "AP50": 98.5, "AP75": 98.5,
                        "AR": 90.0, "AR50": 99.0, "AR75": 99.0, "macro": "PaperDarkPose"}),
])
PUB_SOURCE = "github.com/ostadabbas/Infant-Pose-Estimation README @ a76b4f4 (Test100, GT boxes)"
RUN_RE = re.compile(r"\bR\d{3}[A-Za-z0-9_.\-]*")


# ----------------------------------------------------------------------------- loading
def read_registry(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r.get("run_id")]


def read_prereg(path):
    """RUN_IDs registered for Test100 (any RUN_ID mentioned in a table row of PREREGISTRATION.md)."""
    ids = set()
    if os.path.exists(path):
        for line in open(path):
            if line.strip().startswith("|") and not set(line.strip()) <= set("|-: "):
                ids.update(m.group(0).rstrip(".,;") for m in RUN_RE.finditer(line))
    return ids


def load_metrics(run_id, split, eval_root, runs_root):
    for root, src in ((eval_root, "analysis_eval"), (runs_root, "run_dir")):
        p = os.path.join(root, run_id, f"metrics_{split}.json")
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            if "metrics" in d and "metrics_fixed" in d:     # an eval_coco.py output
                return d, src, p
    return None, None, None


def group_key(row):
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{row.get('model', '')}_{row.get('variant', '')}").strip("_") or row["run_id"]


def macro_name(s):
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
             "8": "Eight", "9": "Nine"}
    out = "".join(words.get(c, c) for c in s)
    parts = [w for w in re.split(r"[^A-Za-z]+", out) if w]
    return "".join(w[0].upper() + w[1:] for w in parts) or "X"


# ----------------------------------------------------------------------------- stats
def mean_std(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else None
    return m, s, len(v)


def pct(x):
    return None if x is None else 100.0 * x


def f1(x, nd=1):
    return "--" if x is None else f"{x:.{nd}f}"


def tex_escape(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#").replace("$", r"\$"))


# ----------------------------------------------------------------------------- build
class Collector:
    def __init__(self, a):
        self.a = a
        self.registry = read_registry(a.registry)
        self.prereg = read_prereg(a.prereg)
        self.configs = json.load(open(a.configs))["configs"] if os.path.exists(a.configs) else {}
        self.run_meta = {r["run_id"]: r for r in self.registry}
        # discover runs that have metrics but no registry row (flagged)
        found = set()
        for root in (a.eval_root, a.runs_root):
            for p in glob.glob(os.path.join(root, "R*", "metrics_*.json")):
                found.add(os.path.basename(os.path.dirname(p)))
        self.unregistered = sorted(found - set(self.run_meta))
        for rid in self.unregistered:
            self.run_meta[rid] = {"run_id": rid, "status": "unregistered", "notes": "metrics found; no registry row"}
        # config groups
        self.groups = OrderedDict()
        assigned = set()
        for key, c in self.configs.items():
            ids = [r for r in c.get("run_ids", []) if r in self.run_meta or True]
            self.groups[key] = dict(c, key=key, run_ids=ids)
            assigned.update(ids)
        for rid, row in self.run_meta.items():
            if rid in assigned:
                continue
            k = group_key(row)
            g = self.groups.setdefault(k, {"key": k, "label": k, "run_ids": [], "role": row.get("stream") or "other",
                                           "what_changed": row.get("variant", ""), "hypothesis": row.get("notes", "")})
            g["run_ids"].append(rid)
        self.single_ci = {}      # (run_id, split) -> (lo, hi) in percent, from bootstrap_ci.py single
        for p in glob.glob(os.path.join(a.tables, "bootstrap_*.json")):
            try:
                d = json.load(open(p))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(d, dict) and d.get("mode") == "single" and d.get("run_id"):
                sp = next((x for x in SPLITS if x in os.path.basename(str(d.get("gt")))), None)
                if sp:
                    lo, hi = d["boot_ci95"]["AP"]
                    self.single_ci[(d["run_id"], sp)] = (100 * lo, 100 * hi, p)
        self.metrics = {}
        for rid in self.run_meta:
            for sp in SPLITS:
                d, src, p = load_metrics(rid, sp, a.eval_root, a.runs_root)
                if d:
                    self.metrics[(rid, sp)] = (d, src, p)
        # test-time ablations: metrics_val400<tag>.json (Val400 only; tagged Test100 files are never read)
        self.tag_metrics = {}
        for root in (a.eval_root, a.runs_root):
            for p in glob.glob(os.path.join(root, "R*", "metrics_val400?*.json")):
                rid = os.path.basename(os.path.dirname(p))
                tag = os.path.basename(p)[len("metrics_val400"):-len(".json")]
                if (rid, tag) in self.tag_metrics:
                    continue
                try:
                    d = json.load(open(p))
                except (OSError, json.JSONDecodeError):
                    continue
                if "metrics" in d and "metrics_fixed" in d:
                    self.tag_metrics[(rid, tag)] = (d, p)

    def test_allowed(self, rid, group=None):
        g = group or next((g for g in self.groups.values() if rid in g["run_ids"]), {})
        return rid in self.prereg or g.get("role") in ("released", "retrain") or bool(g.get("test_allowed"))

    def value(self, rid, split, metric, protocol="paper"):
        x = self.metrics.get((rid, split))
        if not x:
            return None
        d = x[0]["metrics" if protocol == "paper" else "metrics_fixed"]
        return d.get(metric)

    def config_summary(self, g, split, protocol="paper"):
        rids = [r for r in g["run_ids"] if (r, split) in self.metrics]
        if split == "test100":
            rids = [r for r in rids if self.test_allowed(r, g)]
        out = OrderedDict(n_runs=len(rids), run_ids=rids)
        for m in METRICS:
            mu, sd, n = mean_std([self.value(r, split, m, protocol) for r in rids])
            out[m] = pct(mu)
            out[m + "_std"] = pct(sd)
        return out


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, "w").close()
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (";".join(v) if isinstance(v, list) else ("" if v is None else
                        (f"{v:.4f}" if isinstance(v, float) else v))) for k, v in r.items()})


def build(c, a):
    tables = OrderedDict()
    # ---- per-run table (every run, every split) -> seeds / attempts
    per_run = []
    for rid, meta in c.run_meta.items():
        g = next((g for g in c.groups.values() if rid in g["run_ids"]), {})
        row = OrderedDict(run_id=rid, config=g.get("key"), label=g.get("label"), role=g.get("role"),
                          seed=meta.get("seed"), status=meta.get("status"), job_id=meta.get("job_id"),
                          git_sha=meta.get("git_sha"))
        for sp in SPLITS:
            if (rid, sp) not in c.metrics:
                continue
            if sp == "test100" and not c.test_allowed(rid, g):
                row["test100_note"] = "withheld (not pre-registered)"
                continue
            d, src, p = c.metrics[(rid, sp)]
            for m in METRICS:
                row[f"{sp}_{m}"] = pct(d["metrics"].get(m))
                row[f"{sp}_{m}_fixed"] = pct(d["metrics_fixed"].get(m))
            z = d.get("id0_artifact")
            if z:                                   # D13: test0's score rank and the paper-fixed gap
                row[f"{sp}_id0_score_rank"] = z.get("score_rank")
                row[f"{sp}_id0_rank_pct"] = None if z.get("rank_percentile") is None else 100 * z["rank_percentile"]
                row[f"{sp}_paper_minus_fixed_AP"] = pct(z["paper_minus_fixed"]["AP"])
            row[f"{sp}_source"] = p
            row[f"{sp}_evaluator"] = d.get("evaluator")
            row[f"{sp}_pred_sha256"] = d.get("pred", {}).get("sha256")
            row[f"{sp}_clean_test"] = CLEAN[sp]
        per_run.append(row)
    tables["per_run"] = per_run

    # ---- seeds: mean ± std per config
    seeds = []
    for k, g in c.groups.items():
        for sp in ("val400", "test100"):
            s = c.config_summary(g, sp)
            if s["n_runs"] == 0:
                continue
            seeds.append(OrderedDict(config=k, label=g.get("label"), split=sp, split_role=SPLIT_ROLE[sp],
                                     clean_test=CLEAN[sp], n_runs=s["n_runs"], run_ids=s["run_ids"],
                                     **{m: s[m] for m in METRICS}, **{m + "_std": s[m + "_std"] for m in METRICS},
                                     AP_fixed=c.config_summary(g, sp, "fixed")["AP"]))
    tables["seeds"] = seeds

    # ---- attempts: every config, Val400 AP; Test100 only where allowed
    attempts = []
    for i, (k, g) in enumerate(c.groups.items(), 1):
        v = c.config_summary(g, "val400")
        t = c.config_summary(g, "test100")
        attempts.append(OrderedDict(
            attempt=i, config=k, label=g.get("label"), role=g.get("role"), what_changed=g.get("what_changed", ""),
            hypothesis=g.get("hypothesis", ""), n_runs=len(g["run_ids"]), run_ids=g["run_ids"],
            val400_AP=v["AP"], val400_AP_std=v["AP_std"], val400_AP_80_95=v["AP_80_95"],
            test100_AP=t["AP"] if t["n_runs"] else None, test100_AP_std=t["AP_std"] if t["n_runs"] else None,
            test100_AP_fixed=c.config_summary(g, "test100", "fixed")["AP"] if t["n_runs"] else None,
            test100_reported=bool(t["n_runs"]),
            test100_rule="released/retrain (reproduction target)" if g.get("role") in ("released", "retrain") else
            ("pre-registered" if t["n_runs"] else "withheld: selection on Val400 only"),
            status=";".join(sorted({c.run_meta.get(r, {}).get("status", "") or "" for r in g["run_ids"]}))))
    # test-time ablation rows (Val400 only)
    n = len(attempts)
    for k, g in c.groups.items():
        tags = sorted({t for (r, t) in c.tag_metrics if r in g["run_ids"]})
        for tag in tags:
            rids = [r for r in g["run_ids"] if (r, tag) in c.tag_metrics]
            mu, sd, _ = mean_std([c.tag_metrics[(r, tag)][0]["metrics"]["AP"] for r in rids])
            st, _, _ = mean_std([c.tag_metrics[(r, tag)][0]["metrics"]["AP_80_95"] for r in rids])
            n += 1
            attempts.append(OrderedDict(
                attempt=n, config=f"{k}{tag}", label=f"{g.get('label')} [{tag.strip('_')}]", role="test-time ablation",
                what_changed=f"test-time: {tag.strip('_')}", hypothesis=g.get("hypothesis", ""), n_runs=len(rids),
                run_ids=rids, val400_AP=pct(mu), val400_AP_std=pct(sd), val400_AP_80_95=pct(st), test100_AP=None,
                test100_AP_std=None, test100_AP_fixed=None, test100_reported=False,
                test100_rule="withheld: test-time ablation (Val400 only)", status="done"))
    tables["attempts"] = attempts

    # ---- reproduction: paper vs released vs retrains (Test100 + Val400)
    repro = []
    for pk, p in PUBLISHED.items():
        repro.append(OrderedDict(row=p["label"], source=PUB_SOURCE, protocol="paper", n_runs=None, run_ids=[],
                                 **{m: p.get(m) for m in METRICS}, test100_AP_fixed=None, val400_AP=None))
    for k, g in c.groups.items():
        if g.get("role") not in ("released", "retrain"):
            continue
        t, tf, v = c.config_summary(g, "test100"), c.config_summary(g, "test100", "fixed"), c.config_summary(g, "val400")
        ci = c.single_ci.get((t["run_ids"][0], "test100")) if t["n_runs"] == 1 else None
        repro.append(OrderedDict(row=g.get("label"), source="eval/eval_coco.py", protocol="paper",
                                 n_runs=t["n_runs"], run_ids=t["run_ids"], **{m: t[m] for m in METRICS},
                                 AP_std=t["AP_std"], AP_ci_lo=ci[0] if ci else None, AP_ci_hi=ci[1] if ci else None,
                                 test100_AP_fixed=tf["AP"], val400_AP=v["AP"], val400_AP_std=v["AP_std"]))
    tables["reproduction"] = repro

    # ---- main comparison: published, released, control, best candidate(s), ensembles (clean Test100 only)
    main = []
    for pk in ("paper_fidip_darkpose",):
        p = PUBLISHED[pk]
        main.append(OrderedDict(model=p["label"], role="published", run_ids=[], n_runs=None, test100_AP=p["AP"],
                                test100_AP_std=None, test100_AP50=p["AP50"], test100_AP75=p["AP75"], test100_AR=p["AR"],
                                test100_AP_80_95=(10 * p["AP"] - 6 * p["AP50"]) / 4, test100_AP_fixed=None,
                                val400_AP=None, eval="published (paper protocol)"))
    for k, g in c.groups.items():
        if g.get("role") not in ("released", "retrain", "control", "candidate", "final", "ensemble"):
            continue
        t = c.config_summary(g, "test100")
        if not t["n_runs"]:
            continue
        tf, v = c.config_summary(g, "test100", "fixed"), c.config_summary(g, "val400")
        ci = c.single_ci.get((t["run_ids"][0], "test100")) if t["n_runs"] == 1 else None
        main.append(OrderedDict(model=g.get("label"), role=g.get("role"), run_ids=t["run_ids"], n_runs=t["n_runs"],
                                test100_AP=t["AP"], test100_AP_std=t["AP_std"],
                                test100_AP_ci_lo=ci[0] if ci else None, test100_AP_ci_hi=ci[1] if ci else None,
                                test100_AP50=t["AP50"],
                                test100_AP75=t["AP75"], test100_AR=t["AR"], test100_AP_80_95=t["AP_80_95"],
                                test100_AP_fixed=tf["AP"], val400_AP=v["AP"],
                                eval="eval_coco.py v1 (paper protocol; GT boxes; rescored)"))
    tables["main_comparison"] = main

    # ---- ablations: everything that is not a reproduction target (Val400; Test100 only if registered)
    tables["ablations"] = [r for r in attempts if r["role"] not in ("released", "retrain")]

    # ---- bootstrap CIs
    boots = []
    for p in sorted(glob.glob(os.path.join(a.tables, "bootstrap_*.json"))):
        d = json.load(open(p))
        if not isinstance(d, dict) or "mode" not in d:
            continue                      # not a bootstrap_ci.py output
        items = d["contrasts"] if d.get("mode") == "contrasts" else [d]
        for r in items:
            if "delta" not in r:
                continue
            dd = r["delta"].get(r.get("metric", "AP"), r["delta"]["AP"])
            boots.append(OrderedDict(
                file=os.path.relpath(p, REPO) if p.startswith(REPO) else p,
                name=r.get("name") or f"{r['a']['label']} vs {r['b']['label']}", mode=r.get("mode"),
                protocol=r.get("protocol"), metric=r.get("metric", "AP"), n_images=r.get("n_images"), B=r.get("B"),
                a_label=r["a"]["label"], a_run_ids=r["a"].get("run_ids"), a_mean=pct(r["a"]["mean"]["AP"]),
                a_std=pct(r["a"]["std_across_runs"]["AP"]), b_label=r["b"]["label"], b_run_ids=r["b"].get("run_ids"),
                b_mean=pct(r["b"]["mean"]["AP"]), b_std=pct(r["b"]["std_across_runs"]["AP"]),
                delta=pct(dd["observed"]), ci_lo=pct(dd["ci95"][0]), ci_hi=pct(dd["ci95"][1]),
                p_one_sided=r.get("p_one_sided_a_gt_b"), p_holm=r.get("p_holm"),
                seed_pairs_a_gt_b=r.get("seed_pairs_a_gt_b"),
                images_win_tie_loss="/".join(str(r["per_image_consistency"][k]) for k in ("wins", "ties", "losses"))
                if r.get("per_image_consistency") else None,
                sens_protocol=(r.get("sensitivity") or {}).get("protocol"),
                sens_delta=pct(((r.get("sensitivity") or {}).get("delta") or {}).get(r.get("metric", "AP"), {})
                               .get("observed")) if r.get("sensitivity") else None,
                sens_ci_lo=pct(r["sensitivity"]["delta"][r.get("metric", "AP")]["ci95"][0]) if r.get("sensitivity") else None,
                sens_ci_hi=pct(r["sensitivity"]["delta"][r.get("metric", "AP")]["ci95"][1]) if r.get("sensitivity") else None,
                sens_p_holm=(r.get("sensitivity") or {}).get("p_holm"),
                robust_both_protocols=r.get("a_gt_b_robust_both_protocols"),
                robust_claim=r.get("robust_claim"),
                gt=r.get("gt"), self_check_passed=all(x["passed"] for x in r["a"]["self_check"] + r["b"]["self_check"])))
    tables["bootstrap_summary"] = boots
    return tables


# ----------------------------------------------------------------------------- LaTeX
def latex(c, tables, a):
    L = ["% AUTO-GENERATED by scripts/collect_results.py -- do not edit by hand.",
         "% Requires \\usepackage{booktabs}. Numbers are percent (COCO keypoint AP x 100).",
         f"% Published numbers: {PUB_SOURCE}.", ""]
    macros = []

    seen = set()

    def mac(name, val, prov):
        name = macro_name(name)
        base, k = name, 0
        while name in seen:                      # LaTeX refuses duplicate \newcommand
            k += 1
            name = base + "Dup" + "ABCDEFGHIJ"[min(k, 9) - 1]
        seen.add(name)
        macros.append((name, val, prov))
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}  % {prov}")

    for pk, p in PUBLISHED.items():
        for m in ("AP", "AP50", "AP75", "AR"):
            mac(f"{p['macro']} {m.replace('50', ' fifty').replace('75', ' seventyfive')}", f1(p[m]), PUB_SOURCE)
        mac(f"{p['macro']} AP strict", f1((10 * p["AP"] - 6 * p["AP50"]) / 4, 2),
            "implied: (10 AP - 6 AP50)/4, since AP50 = AP75 means AP is flat over .50-.75")
    for k, g in c.groups.items():
        base = g.get("macro") or k
        for sp, tag in (("test100", "Test"), ("val400", "Val")):
            for proto in ("paper", "fixed"):
                s = c.config_summary(g, sp, proto)
                if not s["n_runs"]:
                    continue
                suf = "" if proto == "paper" else " fixed"
                prov = f"{sp} {proto}: " + ",".join(s["run_ids"])
                mac(f"{base} {tag} AP{suf}", f1(s["AP"]), prov)
                mac(f"{base} {tag} AP strict{suf}", f1(s["AP_80_95"]), prov)
                if proto == "paper":
                    mac(f"{base} {tag} AP fifty", f1(s["AP50"]), prov)
                    mac(f"{base} {tag} AP seventyfive", f1(s["AP75"]), prov)
                    mac(f"{base} {tag} AR", f1(s["AR"]), prov)
                    if s["n_runs"] > 1:
                        mac(f"{base} {tag} AP std", f1(s["AP_std"], 2), prov)
                        mac(f"{base} {tag} n runs", str(s["n_runs"]), prov)
    for (rid, sp), (lo, hi, src) in sorted(c.single_ci.items()):
        g = next((g for g in c.groups.values() if rid in g["run_ids"]), None)
        if g is None or (sp == "test100" and not c.test_allowed(rid, g)):
            continue
        tag = {"test100": "Test", "val400": "Val"}.get(sp)
        if tag and len(g["run_ids"]) == 1:
            base = g.get("macro") or g["key"]
            mac(f"{base} {tag} AP ci lo", f1(lo), f"{sp} bootstrap: {rid}")
            mac(f"{base} {tag} AP ci hi", f1(hi), f"{sp} bootstrap: {rid}")
    for b in tables["bootstrap_summary"]:
        nm = b["name"]
        mac(f"Boot {nm} delta", f1(b["delta"], 2), b["file"])
        mac(f"Boot {nm} ci lo", f1(b["ci_lo"], 2), b["file"])
        mac(f"Boot {nm} ci hi", f1(b["ci_hi"], 2), b["file"])
        if b["p_one_sided"] is not None:
            mac(f"Boot {nm} p", f"{b['p_one_sided']:.4f}", b["file"])
        if b["p_holm"] is not None:
            mac(f"Boot {nm} p holm", f"{b['p_holm']:.4f}", b["file"])
        if b.get("sens_delta") is not None:
            mac(f"Boot {nm} fixed delta", f1(b["sens_delta"], 2), b["file"])
            mac(f"Boot {nm} fixed ci lo", f1(b["sens_ci_lo"], 2), b["file"])
            mac(f"Boot {nm} fixed ci hi", f1(b["sens_ci_hi"], 2), b["file"])
            mac(f"Boot {nm} robust", "yes" if b.get("robust_both_protocols") else "no", b["file"])
    L.append("")

    def table(caption, label, header, rows, align):
        L.extend(["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                  f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"])
        for r in rows:
            L.append(" & ".join(r) + " \\\\")
        L.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    def pm(m, s):
        return f1(m) if s is None else f"{f1(m)}\\,$\\pm$\\,{f1(s)}"

    rows = []
    for r in tables["reproduction"]:
        rows.append([tex_escape(r["row"]), str(r["n_runs"] or "--"), pm(r["AP"], r.get("AP_std")), f1(r["AP50"]),
                     f1(r["AP75"]), f1(r["AR"]), f1(r["AP_80_95"]) if r.get("AP_80_95") is not None else
                     f1((10 * r["AP"] - 6 * r["AP50"]) / 4, 2) if r["AP"] is not None and r["AP50"] is not None else "--",
                     f1(r.get("test100_AP_fixed")), pm(r.get("val400_AP"), r.get("val400_AP_std"))])
    table("Reproduction on SyRIP Test100 (GT boxes). Paper protocol unless marked; ``fixed'' removes the "
          "pycocotools annotation-id-0 artefact (Sec.~\\ref{sec:eval}). Mean\\,$\\pm$\\,std over seeds. "
          "AP$_{.80:.95}$ for the paper rows is implied by AP, AP$_{50}$ and AP$_{75}$. Val400 is the selection split.",
          "tab:repro", ["Model", "runs", "AP", "AP$_{50}$", "AP$_{75}$", "AR", "AP$_{.80:.95}$", "AP (fixed)",
                        "Val400 AP"], rows, "lrrrrrrrr")
    rows = []
    for r in tables["main_comparison"]:
        rows.append([tex_escape(r["model"]), str(r["n_runs"] or "--"), pm(r["test100_AP"], r["test100_AP_std"]),
                     f1(r["test100_AP50"]), f1(r["test100_AP75"]), f1(r["test100_AR"]),
                     f1(r["test100_AP_80_95"], 2 if r["role"] == "published" else 1),
                     f1(r["test100_AP_fixed"]), f1(r["val400_AP"])])
    table("Main comparison on the clean Test100 split. Only released/retrained reproduction targets and "
          "pre-registered configurations are evaluated on Test100.", "tab:main",
          ["Model", "runs", "AP", "AP$_{50}$", "AP$_{75}$", "AR", "AP$_{.80:.95}$", "AP (fixed)", "Val400 AP"],
          rows, "lrrrrrrrr")
    rows = []
    for r in tables["attempts"]:
        rows.append([str(r["attempt"]), tex_escape(r["label"] or r["config"]), tex_escape(r["what_changed"] or ""),
                     str(r["n_runs"]), pm(r["val400_AP"], r["val400_AP_std"]),
                     pm(r["test100_AP"], r["test100_AP_std"]) if r["test100_reported"] else "withheld"])
    table("Every attempt and its accuracy. Val400 AP drives every decision. Test100 is shown only for "
          "reproduction targets and pre-registered finalists (``withheld'' otherwise).", "tab:attempts",
          ["\\#", "Configuration", "What changed", "runs", "Val400 AP", "Test100 AP"], rows, "rllrrr")
    if tables["bootstrap_summary"]:
        rows = []
        for b in tables["bootstrap_summary"]:
            fx = (f"{b['sens_delta']:+.2f} [{b['sens_ci_lo']:+.2f}, {b['sens_ci_hi']:+.2f}]"
                  if b.get("sens_delta") is not None else "--")
            rows.append([tex_escape(b["name"]), f"{b['delta']:+.2f}", f"[{b['ci_lo']:+.2f}, {b['ci_hi']:+.2f}]",
                         "--" if b["p_one_sided"] is None else f"{b['p_one_sided']:.4f}",
                         "--" if b["p_holm"] is None else f"{b['p_holm']:.4f}", fx,
                         {True: "yes", False: "no"}.get(b.get("robust_both_protocols"), "--"),
                         tex_escape(b["images_win_tie_loss"] or "--")])
        table("Paired image-level bootstrap ($B$ resamples) of $\\Delta$AP on Test100 (paper protocol; one-sided $p$ "
              "for $H_1$: A $>$ B, Holm-adjusted over the declared contrasts), the same contrast under the fixed "
              "protocol, and whether the 95\\% CI excludes 0 under both (D13). Last column: per-image OKS wins/ties/losses.",
              "tab:boot", ["Contrast", "$\\Delta$AP", "95\\% CI", "$p$", "$p_{\\mathrm{Holm}}$", "fixed: $\\Delta$ [CI]",
                           "robust", "W/T/L"], rows, "lrrrrrcr")
    return "\n".join(L) + "\n", macros


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=os.path.join(REPO, "results", "registry.csv"))
    ap.add_argument("--prereg", default=os.path.join(REPO, "results", "PREREGISTRATION.md"))
    ap.add_argument("--configs", default=os.path.join(REPO, "results", "configs.json"))
    ap.add_argument("--eval-root", default=os.path.join(REPO, "results", "analysis_eval"))
    ap.add_argument("--runs-root", default=os.path.join(REPO, "results", "cluster_runs"))
    ap.add_argument("--tables", default=os.path.join(REPO, "results", "tables"))
    ap.add_argument("--tex", default=os.path.join(REPO, "report", "generated", "results_tables.tex"))
    a = ap.parse_args(argv)
    c = Collector(a)
    tables = build(c, a)
    for name, rows in tables.items():
        write_csv(os.path.join(a.tables, f"{name}.csv"), rows)
        with open(os.path.join(a.tables, f"{name}.json"), "w") as f:
            json.dump(rows, f, indent=1, default=str)
    tex, macros = latex(c, tables, a)
    os.makedirs(os.path.dirname(a.tex), exist_ok=True)
    with open(a.tex, "w") as f:
        f.write(tex)
    with open(os.path.join(os.path.dirname(a.tex), "MACROS.md"), "w") as f:
        f.write("# Macros in results_tables.tex (auto-generated; use these, never hand-typed numbers)\n\n")
        f.write("| Macro | Value | Provenance |\n|---|---|---|\n")
        for n, v, p in macros:
            f.write(f"| `\\{n}` | {v} | {p} |\n")
    print(f"[collect] {len(c.run_meta)} runs ({len(c.unregistered)} unregistered), {len(c.groups)} configs, "
          f"{len(c.metrics)} metric files, {len(tables['bootstrap_summary'])} bootstrap results, prereg={sorted(c.prereg)}")
    print(f"  -> {a.tables}/*.csv|json, {a.tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
