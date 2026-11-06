"""report_interface.py -- fills the report stream's generated-file interface (called by collect_results.py).

Writes, idempotently:
- report/generated/numbers.tex
  - Every `\\setres{key}{value}` line already present keeps its key, order and comment; only the value changes.
  - Unset keys stay `\\tbd`.
- report/generated/tab_<name>.tex
  - One bare tabular per file (no float/caption/label), with the placeholder's column layout.
  - hparams/compute/patches are written only when their data source exists, so pre-filled placeholders survive.
- results/tables/provenance.csv
  - key, value, run_ids, metrics_file, protocol, split, for every value set.
  - MANIFEST §8 is generated from it.
- report/figures/generated/<name>.{pdf,png}: copies of the analysis figures under the report's fixed names.

Formats (numbers.tex interface):
- AP: percent, 1 decimal.
- signed differences: $+0.8$.
- p-values: 3 decimals or $<$0.001.
- SHAs: \\texttt{abc1234}.
Test100 values exist only for released/retrain/pre-registered runs (Collector.test_allowed).
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import shutil
from collections import OrderedDict

PUBLISHED_AP = 93.6
TBD = r"\tbd"


# ----------------------------------------------------------------------------- formatting
def pct1(x):
    return None if x is None else f"{100 * x:.1f}"


def num1(x):
    return None if x is None else f"{x:.1f}"


def signed(x, nd=1):
    if x is None:
        return None
    return f"${'+' if x >= 0 else '-'}{abs(x):.{nd}f}$"


def pval(p):
    if p is None:
        return None
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def pm(mean, sd):
    if mean is None:
        return TBD
    return f"{mean:.1f}" if not sd else f"{mean:.1f}\\,{{\\scriptsize$\\pm${sd:.1f}}}"


def tex(s):
    """Plain text -> LaTeX: escape specials, then a few safe typographic substitutions."""
    s = str(s)
    s = s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    s = s.replace("$", r"\$")
    s = s.replace("^", r"\^{}").replace("~", r"\~{}")
    s = s.replace("+-", r"$\pm$").replace("->", r"$\to$")
    s = re.sub(r"\blambda\b", r"$\\lambda$", s)
    return re.sub(r"(\d+)x(\d+)", r"\1$\\times$\2", s)


def v(x):
    return TBD if x is None else x


def find_tables(root, pattern):
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- the writer
class ReportInterface:
    def __init__(self, c, tables, repo, tables_dir):
        self.c, self.t, self.repo, self.tdir = c, tables, repo, tables_dir
        self.values = OrderedDict()          # key -> (value, run_ids, file, protocol, split)
        self.cfg = c.groups

    # ---- data access
    def group(self, key=None, role=None):
        if key and key in self.cfg:
            return self.cfg[key]
        if role:
            return next((g for g in self.cfg.values() if g.get("role") == role), None)
        return None

    def m(self, rid, split, proto="paper"):
        x = self.c.metrics.get((rid, split))
        if not x:
            return None, None
        if split in ("test100", "test100_clean", "validate_infant", "validate500") and not self.c.test_allowed(rid):
            return None, None
        d, src, path = x
        return d, path

    def metric(self, rid, split, name, proto="paper"):
        d, path = self.m(rid, split)
        if not d:
            return None, None
        return d["metrics" if proto == "paper" else "metrics_fixed"].get(name), path

    def set(self, key, value, run_ids="", path="", protocol="", split=""):
        if value is None:
            return
        self.values[key] = (value, ";".join(run_ids) if isinstance(run_ids, (list, tuple)) else run_ids,
                            path or "", protocol, split)

    # ---- scalars
    def fill_scalars(self):
        c = self.c
        for gkey, prefix in (("rel_fidip_w48", "rel.fidip"), ("rel_fidip_mobile", "rel.mobile")):
            g = self.group(gkey)
            if not g or not g["run_ids"]:
                continue
            rid = g["run_ids"][0]
            for name, k in (("AP", "ap"), ("AP50", "ap50"), ("AP75", "ap75"), ("AR", "ar"), ("AP_80_95", "apstrict")):
                val, path = self.metric(rid, "test100", name)
                if prefix == "rel.mobile" and k != "ap":
                    continue
                self.set(f"{prefix}.test.{k}", pct1(val), [rid], path, "paper", "test100")
            for name, k in (("AP", "ap.fixed"), ("AR", "ar.fixed")):
                val, path = self.metric(rid, "test100", name, "fixed")
                if prefix == "rel.fidip":
                    self.set(f"{prefix}.test.{k}", pct1(val), [rid], path, "fixed", "test100")
            ap, path = self.metric(rid, "test100", "AP")
            if prefix == "rel.fidip" and ap is not None:
                self.set("rel.fidip.test.diffpub", signed(100 * ap - PUBLISHED_AP), [rid], path, "paper", "test100")
                d, _ = self.m(rid, "test100")
                z = (d or {}).get("id0_artifact") or {}
                if z.get("score_rank") is not None:
                    self.set("rel.fidip.test0rank", str(z["score_rank"]), [rid], path, "paper", "test100")
                cl, cpath = self.metric(rid, "test100_clean", "AP")
                self.set("rel.fidip.test.clean.ap", pct1(cl), [rid], cpath, "paper", "test100_clean")
                up = self._upstream(rid, "test100")
                if up is not None:
                    self.set("rel.fidip.test.ap.upstream", f"{100 * up:.1f}", [rid],
                             os.path.join("runs", rid, "metrics_upstream_test100.txt"), "paper (upstream code)", "test100")
            va, vpath = self.metric(rid, "val400", "AP")
            self.set(f"{prefix}.val.ap", pct1(va), [rid], vpath, "paper", "val400")
        # zero-shot DarkPose (COCO only), if repro registered it
        g = self.group("zs_darkpose")
        if g and g["run_ids"]:
            rid = g["run_ids"][0]
            for split, k in (("test100", "zs.darkpose.test.ap"), ("val400", "zs.darkpose.val.ap")):
                val, path = self.metric(rid, split, "AP")
                self.set(k, pct1(val), [rid], path, "paper", split)
        # retrains
        for gkey, prefix in (("rt_fidip", "rt.fidip"), ("rt_ft", "rt.ft")):
            g = self.group(gkey)
            if not g:
                continue
            for split, tag in (("test100", "test"), ("val400", "val")):
                s = c.config_summary(g, split)
                if s["n_runs"]:
                    self.set(f"{prefix}.{tag}.ap.mean", num1(s["AP"]), s["run_ids"], "", "paper", split)
                    self.set(f"{prefix}.{tag}.ap.sd", num1(s["AP_std"]), s["run_ids"], "", "paper", split)
            fin = [self.c.final_metrics[(r, "test100")][0]["metrics"]["AP"] for r in g["run_ids"]
                   if (r, "test100") in self.c.final_metrics and self.c.test_allowed(r)]
            if fin:                                    # (b) final-epoch checkpoint, also pre-registered
                mu = sum(fin) / len(fin)
                sd = (sum((x - mu) ** 2 for x in fin) / (len(fin) - 1)) ** 0.5 if len(fin) > 1 else None
                ids = [r for r in g["run_ids"] if (r, "test100") in self.c.final_metrics]
                self.set(f"{prefix}.test.ap.final.mean", pct1(mu), ids, "metrics_test100_final.json", "paper", "test100")
                self.set(f"{prefix}.test.ap.final.sd", pct1(sd) if sd is not None else "---", ids,
                         "metrics_test100_final.json", "paper", "test100")
            have = [r for r in g.get("run_ids", []) if (r, "val400") in c.metrics or (r, "test100") in c.metrics]
            if gkey == "rt_fidip" and have:          # seeds that actually have results (FiDIP arm)
                self.set("rt.seeds", str(len(have)), have, "results/analysis_eval", "", "")
            if g.get("epochs"):
                self.set("rt.epochs", str(g["epochs"]), g["run_ids"], "results/configs.json", "", "")
        b = self._boot("rt_fidip", "rt_ft")
        if b:
            self.set("rt.delta.test.ap", signed(b["delta"]), b["ids"], b["file"], b["protocol"], "test100")
            self.set("rt.delta.test.ci.lo", signed(b["ci_lo"]), b["ids"], b["file"], b["protocol"], "test100")
            self.set("rt.delta.test.ci.hi", signed(b["ci_hi"]), b["ids"], b["file"], b["protocol"], "test100")
            self.set("rt.delta.test.p", pval(b["p"]), b["ids"], b["file"], b["protocol"], "test100")
        # improvement study
        g = self.group(role="control")
        if g:
            s = c.config_summary(g, "val400")
            self.set("imp.control.name", tex(g.get("label", g["key"])), g["run_ids"], "results/configs.json", "", "")
            if s["n_runs"]:
                self.set("imp.control.val.ap.mean", num1(s["AP"]), s["run_ids"], "", "paper", "val400")
                self.set("imp.control.val.ap.sd", num1(s["AP_std"]) if s["AP_std"] is not None else "---",
                         s["run_ids"], "", "paper", "val400")
        self._fill_best()
        ge = self.group(role="ensemble")
        if ge:
            st = c.config_summary(ge, "test100")
            if st["n_runs"]:
                self.set("ens.name", tex(ge.get("label", ge["key"])), st["run_ids"], "results/configs.json")
                self.set("ens.test.ap", num1(st["AP"]), st["run_ids"], "", "paper", "test100")
            b = self._boot(ge["key"], "rel_fidip_w48")
            if b:
                self.set("ens.delta.ci.lo", signed(b["ci_lo"]), b["ids"], b["file"], "paper", "test100")
                self.set("ens.delta.ci.hi", signed(b["ci_hi"]), b["ids"], b["file"], "paper", "test100")
        # bootstrap resamples
        Bs = {b.get("B") for b in self.t.get("bootstrap_summary", []) if b.get("B")}
        self.set("boot.n", str(max(Bs)) if Bs else "10000", "", "bootstrap_ci.py" if not Bs else "results/tables/bootstrap_*.json")
        # attempts
        att = [r for r in self.t["attempts"] if r["role"] != "smoke"]
        self.set("att.n", str(len(att)), "", "results/tables/attempts.csv")
        ctrl = self.group(role="control")
        if ctrl:
            cm = c.config_summary(ctrl, "val400")["AP"]
            if cm is not None:
                nb = sum(1 for r in att if r["val400_AP"] is not None and r["role"] not in ("control", "smoke")
                         and r["val400_AP"] > cm)
                self.set("att.nbetter", str(nb), ctrl["run_ids"], "results/tables/attempts.csv", "paper", "val400")
        self._fill_sensitivity()
        self._fill_val_vs_test()

    def _upstream(self, rid, split):
        for p in find_tables(self.tdir, "crosscheck_upstream.csv"):
            for r in read_csv(p):
                if r["run_id"] == rid and r["split"] == split:
                    return fnum(r["upstream_AP"])
        return None

    def _boot(self, key_a, key_b, protocol=None):
        ga, gb = self.group(key_a), self.group(key_b)
        if not ga or not gb:
            return None
        for b in self.t.get("bootstrap_summary", []):
            if "Test100-clean" in str(b.get("name")):     # headline keys use the official Test100 contrast
                continue
            ia, ib = set(b.get("a_run_ids") or []), set(b.get("b_run_ids") or [])
            if ia and ib and ia <= set(ga["run_ids"]) and ib <= set(gb["run_ids"]):
                return {"delta": b["delta"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                        "p": b["p_holm"] if b.get("p_holm") is not None else b["p_one_sided"],
                        "sens_delta": b.get("sens_delta"), "sens_ci_lo": b.get("sens_ci_lo"),
                        "sens_ci_hi": b.get("sens_ci_hi"), "robust": b.get("robust_both_protocols"),
                        "ids": sorted(ia | ib), "file": b["file"], "protocol": b.get("protocol") or "paper"}
        return None

    def _fill_best(self):
        c = self.c
        g = self.group(role="final")
        if not g:
            return
        ids = g["run_ids"]
        self.set("best.name", tex(g.get("label", g["key"])), ids, "results/configs.json")
        for f in ("input", "flip", "ckptrule"):
            if g.get(f):
                self.set(f"best.{f}", tex(g[f]), ids, "results/configs.json")
        if g.get("prereg_commit"):
            self.set("best.prereg.commit", f"\\texttt{{{g['prereg_commit'][:7]}}}", ids, "results/PREREGISTRATION.md")
        sv, st, sf = c.config_summary(g, "val400"), c.config_summary(g, "test100"), c.config_summary(g, "test100", "fixed")
        self.set("best.val.ap.mean", num1(sv["AP"]), sv["run_ids"], "", "paper", "val400")
        self.set("best.val.ap.sd", num1(sv["AP_std"]), sv["run_ids"], "", "paper", "val400")
        if st["n_runs"]:
            self.set("best.seeds", str(st["n_runs"]), st["run_ids"], "", "paper", "test100")
            for k, name in (("test.ap.mean", "AP"), ("test.ap50", "AP50"), ("test.ap75", "AP75"), ("test.ar", "AR"),
                            ("test.apstrict", "AP_80_95")):
                self.set(f"best.{k}", num1(st[name]), st["run_ids"], "", "paper", "test100")
            self.set("best.test.ap.sd", num1(st["AP_std"]), st["run_ids"], "", "paper", "test100")
            self.set("best.test.ap.fixed", num1(sf["AP"]), sf["run_ids"], "", "fixed", "test100")
            self.set("best.vs.published", signed(st["AP"] - PUBLISHED_AP), st["run_ids"], "", "paper", "test100")
            cl = c.config_summary(g, "test100_clean")
            self.set("best.test.clean.ap", num1(cl["AP"]), cl["run_ids"], "", "paper", "test100_clean")
            d, path = self.m(st["run_ids"][0], "test100")
            z = (d or {}).get("id0_artifact") or {}
            if z.get("score_rank") is not None:
                self.set("best.test0rank", str(z["score_rank"]), [st["run_ids"][0]], path, "paper", "test100")
        b = self._boot(g["key"], "rel_fidip_w48")
        if b:
            self.set("best.delta.ap", signed(b["delta"]), b["ids"], b["file"], "paper", "test100")
            self.set("best.delta.ci.lo", signed(b["ci_lo"]), b["ids"], b["file"], "paper", "test100")
            self.set("best.delta.ci.hi", signed(b["ci_hi"]), b["ids"], b["file"], "paper", "test100")
            self.set("best.delta.p", pval(b["p"]), b["ids"], b["file"], "paper", "test100")
            if b.get("sens_delta") is not None:
                self.set("best.delta.fixed", signed(b["sens_delta"]), b["ids"], b["file"], "fixed", "test100")
                self.set("best.delta.fixed.ci.lo", signed(b["sens_ci_lo"]), b["ids"], b["file"], "fixed", "test100")
                self.set("best.delta.fixed.ci.hi", signed(b["sens_ci_hi"]), b["ids"], b["file"], "fixed", "test100")
                self.set("best.robust", "yes" if b.get("robust") else "no", b["ids"], b["file"], "both", "test100")

    def _fill_sensitivity(self):
        g = self.group("rel_fidip_w48")
        if not g or not g["run_ids"]:
            return
        rid = g["run_ids"][0]
        ap, path = self.metric(rid, "test100", "AP")
        apf, _ = self.metric(rid, "test100", "AP", "fixed")
        if ap is not None and apf is not None:
            self.set("sens.id0.delta", signed(100 * (apf - ap)), [rid], path, "fixed-paper", "test100")
        for p in find_tables(self.tdir, f"id0_rank_sensitivity_{rid}.csv"):
            gaps = [100 * (fnum(r["AP_fixed"]) - fnum(r["AP_paper"])) for r in read_csv(p)
                    if r["rank"] != "summary" and fnum(r["AP_fixed"]) is not None]
            if gaps:
                self.set("sens.id0.min", num1(min(gaps)), [rid], p, "fixed-paper over test0 ranks", "test100")
                self.set("sens.id0.max", num1(max(gaps)), [rid], p, "fixed-paper over test0 ranks", "test100")
        t11, t11p = self.metric(rid, "validate_infant", "AP")
        self.set("sens.test11.ap", pct1(t11), [rid], t11p, "paper", "test100 GT, syrip_for_train images")
        cl, clp = self.metric(rid, "test100_clean", "AP")
        if cl is not None and ap is not None:
            self.set("sens.clean.delta", signed(100 * (cl - ap)), [rid], clp, "paper", "test100_clean - test100")
        for p in find_tables(self.tdir, "gtbox_v100_vs_v500.csv"):
            if rid not in p:
                continue
            for r in read_csv(p):
                if r["pred_boxes"] == "v100 boxes" and r["gt"] == "v100 GT":
                    self.set("sens.box.v100.ap", pct1(fnum(r["AP_paper"])), [rid], p, "paper", "test100 (v100 GT)")
                if r["pred_boxes"] == "v500 boxes" and r["gt"] == "v500 GT":
                    self.set("sens.box.v500.ap", pct1(fnum(r["AP_paper"])), [rid], p, "paper", "test100 ids (v500 GT)")
                if r["pred_boxes"].startswith("box-area ratio"):
                    self.set("data.boxratio.pct", f"{100 * (fnum(r['AP_paper']) - 1):.0f}", "", p, "", "test100 ids")
        for p in find_tables(self.tdir, f"sensitivity_extra_{rid}.json"):
            d = json.load(open(p))
            if d.get("val400_AP_const_area") is not None:
                self.set("sens.area.const.ap", pct1(d["val400_AP_const_area"]), [rid], p, "paper", "val400 (constant area)")
            if d.get("test100_AP_xtcocotools") is not None and ap is not None:   # vs the official eval_coco number
                self.set("sens.xtcoco.delta", signed(100 * (d["test100_AP_xtcocotools"] - ap)),
                         [rid], p, "xtcocotools - paper", "test100")

    def _fill_val_vs_test(self):
        pts = []
        for rid in self.c.run_meta:
            t, _ = self.metric(rid, "test100", "AP")
            va, _ = self.metric(rid, "val400", "AP")
            if t is not None and va is not None:
                pts.append((rid, 100 * va, 100 * t))
        if len(pts) >= 2:
            ids = [p[0] for p in pts]
            self.set("vt.n", str(len(pts)), ids, "results/tables/per_run.csv", "paper", "val400+test100")
            self.set("vt.gap.mean", signed(sum(p[2] - p[1] for p in pts) / len(pts)), ids,
                     "results/tables/per_run.csv", "paper", "test100-val400")
        if len(pts) >= 3:
            def rank(x):
                o = sorted(range(len(x)), key=lambda i: x[i])
                r = [0] * len(x)
                for k, i in enumerate(o):
                    r[i] = k
                return r
            a, b = rank([p[1] for p in pts]), rank([p[2] for p in pts])
            n = len(pts)
            rho = 1 - 6 * sum((x - y) ** 2 for x, y in zip(a, b)) / (n * (n * n - 1))
            self.set("vt.spearman", f"{rho:.2f}", [p[0] for p in pts], "results/tables/per_run.csv", "paper", "")
        self.vt_points = pts

    # ---- numbers.tex
    def write_numbers(self, path):
        lines = open(path).read().splitlines() if os.path.exists(path) else []
        out, seen = [], set()
        pat = re.compile(r"^(\s*\\setres\{([^}]*)\}\{)(.*?)(\}\s*(%.*)?)$")
        for ln in lines:
            m = pat.match(ln)
            if m:
                key = m.group(2)
                seen.add(key)
                val = self.values.get(key, (TBD,))[0]
                comment = m.group(5) or ""
                out.append(f"\\setres{{{key}}}{{{val}}}" + (f"{' ' * max(1, 40 - len(key) - len(val))}{comment}" if comment else ""))
            elif "PLACEHOLDER" in ln:
                out.append(ln.replace("PLACEHOLDER.", "GENERATED by scripts/collect_results.py (do not edit; rerun it).")
                           .replace("PLACEHOLDER", "GENERATED"))
            else:
                out.append(ln)
        extra = [k for k in self.values if k not in seen]
        if extra:
            out.append("% ---- values computed by collect_results.py for keys not (yet) in the documented interface ----")
            out += [f"\\setres{{{k}}}{{{self.values[k][0]}}}" for k in extra]
        with open(path, "w") as f:
            f.write("\n".join(out) + "\n")
        return len(self.values), len(seen)

    def write_provenance(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "value", "run_ids", "metrics_file", "protocol", "split"])
            for k, (val, ids, p, proto, split) in self.values.items():
                w.writerow([k, val, ids, os.path.relpath(p, self.repo) if p and os.path.isabs(p) and
                            p.startswith(self.repo) else p, proto, split])

    # ---- tables
    def _tabular(self, path, header_comment, colspec, header_rows, body_rows):
        L = [f"% {os.path.basename(path)} -- GENERATED by scripts/collect_results.py (do not edit; rerun it)."]
        L += [f"% {h}" for h in header_comment]
        L += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"] + header_rows + ["\\midrule"]
        for r in body_rows:
            L.append(r if r in ("\\midrule",) else " & ".join(r) + "\\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        with open(path, "w") as f:
            f.write("\n".join(L) + "\n")

    def write_tables(self, gen_dir):
        c = self.c
        os.makedirs(gen_dir, exist_ok=True)
        written = []

        def rel_row(label, weights, gkey, pub):
            g = self.group(gkey)
            if not g or not g["run_ids"]:
                return [label, weights, pub] + [TBD] * 9
            rid = g["run_ids"][0]
            d, _ = self.m(rid, "test100")
            dv, _ = self.m(rid, "val400")
            dc, _ = self.m(rid, "test100_clean")
            val = v(pct1(dv["metrics"]["AP"])) if dv else TBD
            if not d:
                return [label, weights, pub] + [TBD] * 8 + [val]
            mm, mf = d["metrics"], d["metrics_fixed"]
            return [label, weights, pub, pct1(mm["AP"]), pct1(mm["AP50"]), pct1(mm["AP75"]), pct1(mm["AP_80_95"]),
                    pct1(mm["AR"]), pct1(mf["AP"]), v(pct1(dc["metrics"]["AP"])) if dc else TBD,
                    v(pct1(dc["metrics_fixed"]["AP"])) if dc else TBD, val]

        def rt_row(label, gkey, pub, variant=None):
            g = self.group(gkey)
            if not g:
                return [label, f"retrain, {TBD}{{}} seeds", pub] + [TBD] * 9
            sm = lambda sp, pr="paper": c.config_summary(g, sp, pr, variant)
            st, sf, sv = sm("test100"), sm("test100", "fixed"), sm("val400")
            sc, scf = sm("test100_clean"), sm("test100_clean", "fixed")
            n = st["n_runs"] or sv["n_runs"]
            rule = "final epoch" if variant == "final" else "best-Val400 epoch"
            return [label, f"retrain, {n or TBD} seeds, {rule}", pub, pm(st["AP"], st["AP_std"]), v(num1(st["AP50"])),
                    v(num1(st["AP75"])), v(num1(st["AP_80_95"])), v(num1(st["AR"])), pm(sf["AP"], sf["AP_std"]),
                    pm(sc["AP"], sc["AP_std"]), pm(scf["AP"], scf["AP_std"]), pm(sv["AP"], sv["AP_std"])]

        p = os.path.join(gen_dir, "tab_reproduction.tex")
        self._tabular(p, ["Test100 = paper protocol unless noted; retrain rows mean +- sd over seeds; both checkpoint "
                          "rules of PREREGISTRATION repro-1.",
                          "Test100-clean = Test100 minus the 5 copies of training images (D16)."], "@{}llcccccccccc@{}",
                      ["& & \\multicolumn{9}{c}{SyRIP Test100} & Val400\\\\",
                       "\\cmidrule(lr){3-11}\\cmidrule(l){12-12}",
                       "Model & Weights / runs & Publ.\\ AP & AP & AP$_{50}$ & AP$_{75}$ & AP$_{80:95}$ & AR & "
                       "AP$^{\\text{fixed}}$ & AP$^{\\text{clean}}$ & AP$^{\\text{clean,fixed}}$ & AP\\\\"],
                      [rel_row("DarkPose W48 (COCO only)", "\\texttt{w48\\_384x288.pth}", "zs_darkpose", "88.5"),
                       rel_row("DarkPose + FiDIP", "\\texttt{hrnet\\_fidip.pth}", "rel_fidip_w48", "93.6"),
                       rel_row("Pose-MobileNet + FiDIP", "\\texttt{mobile\\_fidip.pth}", "rel_fidip_mobile", "79.2"),
                       "\\midrule",
                       rt_row("DarkPose + Finetune ($\\lambda{=}0$)", "rt_ft", "92.7"),
                       rt_row("", "rt_ft", "", variant="final"),
                       rt_row("DarkPose + FiDIP ($\\lambda{=}5{\\cdot}10^{-4}$)", "rt_fidip", "93.6"),
                       rt_row("", "rt_fidip", "", variant="final")])
        written.append(p)

        # final comparison
        def final_row(label, gkey, ref=False):
            g = self.group(gkey) if gkey else None
            if not g:
                return [label] + [TBD] * 8
            st = c.config_summary(g, "test100")
            if not st["n_runs"]:
                return [label] + [TBD] * 8
            seeds = "---" if g.get("role") == "released" else str(st["n_runs"])
            b = None if ref else self._boot(gkey, "rel_fidip_w48")
            dcol = "0 (reference)" if ref else (f"{signed(b['delta'])} [{signed(b['ci_lo'])}, {signed(b['ci_hi'])}]" if b else TBD)
            return [label, seeds, pm(st["AP"], st["AP_std"]), v(num1(st["AP50"])), v(num1(st["AP75"])),
                    v(num1(st["AP_80_95"])), v(num1(st["AR"])), dcol, "---" if ref else (v(pval(b["p"])) if b else TBD)]
        best = self.group(role="final")
        p = os.path.join(gen_dir, "tab_final.tex")
        self._tabular(p, ["Final Test100 comparison (paper protocol); Delta/CI from bootstrap_ci.py vs R001."],
                      "@{}lcccccccc@{}",
                      ["Model & Seeds & AP & AP$_{50}$ & AP$_{75}$ & AP$_{80:95}$ & AR & $\\Delta$AP vs.\\ released [95\\% CI] & $p$\\\\"],
                      [["DarkPose + FiDIP, published", "---", "93.6", "98.5", "98.5", "---", "94.6", "---", "---"],
                       final_row("DarkPose + FiDIP, released ckpt", "rel_fidip_w48", ref=True),
                       final_row("DarkPose + FiDIP, our retrain", "rt_fidip"),
                       "\\midrule",
                       final_row(f"Best configuration ({tex(best['label']) if best else TBD})", best["key"] if best else None)])
        written.append(p)

        # attempts
        p = os.path.join(gen_dir, "tab_attempts.tex")
        rows, last_stream = [], None
        ctrl = self.group(role="control")
        cmean = c.config_summary(ctrl, "val400")["AP"] if ctrl else None
        relv = self.group("rel_fidip_w48")
        rmean = c.config_summary(relv, "val400")["AP"] if relv else None
        att = sorted([r for r in self.t["attempts"] if r["role"] != "smoke"],
                     key=lambda r: min(int(x[1:4]) for x in r["run_ids"]) if r["run_ids"] else 9999)
        for r in att:
            stream = "repro" if any(x.startswith("R0") for x in r["run_ids"]) else "improve"
            if r["role"] in ("ablation",) and stream == "repro":
                stream = "repro (DA)"
            if last_stream is not None and stream != last_stream:
                rows.append("\\midrule")
            last_stream = stream
            gcfg = self.cfg.get(r["config"].split("_noflip")[0], {}) if r.get("config") else {}
            bkey = gcfg.get("baseline")              # Delta vs the config's own baseline when it has one
            if bkey and self.group(bkey):
                ref = c.config_summary(self.group(bkey), "val400")["AP"]
            else:
                ref = cmean if stream == "improve" else rmean
            d = None if (r["val400_AP"] is None or ref is None or r["role"] in ("control", "screen_control", "released",
                                                                                 "retrain", "failed")) \
                else r["val400_AP"] - ref
            outcome = gcfg.get("outcome") or {"released": "reproduction target", "retrain": "reproduction",
                                              "control": "control", "screen_control": "screening control",
                                              "zeroshot": "zero-shot reference", "failed": "failed"}.get(r["role"])
            if outcome is None:
                what = "baseline" if bkey else "control"
                outcome = TBD if d is None else (f"above {what}" if d > 0 else f"not above {what}")
            nums = sorted({x[:4] for x in r["run_ids"]})
            idtxt = (nums[0] if len(nums) == 1 else f"{nums[0]}--{nums[-1]}") if nums else "---"
            rows.append([idtxt, stream, tex(r["what_changed"] or ""), str(r["n_runs"]),
                         "---" if r["role"] == "failed" else pm(r["val400_AP"], r["val400_AP_std"]),
                         v(signed(d)) if d is not None else "---", pm(r["test100_AP"], r["test100_AP_std"])
                         if r["test100_reported"] else "---", outcome])
        self._tabular(p, ["Every attempt (chronological by RUN_ID), Val400 selection; Test100 only for reproduction targets / "
                          "pre-registered.", "Delta Val400: vs the configuration's own baseline (configs.json 'baseline'); "
                          "otherwise improve rows vs the MMPose control, repro rows vs released FiDIP."],
                      "@{}lllcccc l@{}",
                      ["Run ID(s) & Stream & Change vs.\\ control & Seeds & Val400 AP & $\\Delta$ Val400 & Test100 AP & Outcome\\\\"],
                      rows or [[TBD] * 8])
        written.append(p)

        # bootstrap contrasts
        p = os.path.join(gen_dir, "tab_boot.tex")
        brows = []
        for b in self.t.get("bootstrap_summary", []):
            fx = (f"{signed(b['sens_delta'])} [{signed(b['sens_ci_lo'])}, {signed(b['sens_ci_hi'])}]"
                  if b.get("sens_delta") is not None else "---")
            brows.append([tex(b["name"]), signed(b["delta"]), f"[{signed(b['ci_lo'])}, {signed(b['ci_hi'])}]",
                          v(pval(b["p_one_sided"])), v(pval(b["p_holm"])), fx,
                          {True: "yes", False: "no"}.get(b.get("robust_both_protocols"), "---"),
                          tex(b.get("images_win_tie_loss") or "---")])
        if not brows:
            brows = [["best vs.\\ released FiDIP"] + [TBD] * 7, ["FiDIP retrain vs.\\ fine-tune"] + [TBD] * 7]
        self._tabular(p, ["Paired image-level bootstrap on Test100 under both protocols (D13)."], "@{}lrrrrrcr@{}",
                      ["Contrast & $\\Delta$AP & 95\\% CI & $p$ & $p_{\\mathrm{Holm}}$ & fixed: $\\Delta$ [CI] & robust & W/T/L\\\\"],
                      brows)
        written.append(p)

        # backbone (controlled comparison)
        p = os.path.join(gen_dir, "tab_backbone.tex")
        bb = []
        for key, name, inp in (("mmp_hrnet_base", "HRNet-W48 + DARK (control)", "384$\\times$288"),
                               ("mmp_vitl256_base", "ViTPose-L", "256$\\times$192"),
                               ("mmp_vitl384_base", "ViTPose-L", "384$\\times$288"),
                               ("mmp_vith256_base", "ViTPose-H (selected)", "256$\\times$192"),
                               ("mmp_vith384_base", "ViTPose-H", "384$\\times$288")):
            g = self.group(key)
            if not g:
                bb.append([name, inp] + [TBD] * 4)
                continue
            sv = c.config_summary(g, "val400")
            st = c.config_summary(g, "test100")
            bb.append([name, inp, pm(sv["AP"], sv["AP_std"]), v(num1(sv["AP_80_95"])),
                       str(sv["n_runs"]) if sv["n_runs"] else TBD, pm(st["AP"], st["AP_std"]) if st["n_runs"] else "---"])
        self._tabular(p, ["Controlled backbone comparison under one MMPose recipe (Val400 selection)."], "@{}llcccc@{}",
                      ["Backbone & Input & Val400 AP & Val400 AP$_{80:95}$ & Seeds & Test100 AP\\\\"], bb)
        written.append(p)

        # ablations: configs with factor/setting/baseline
        p = os.path.join(gen_dir, "tab_ablations.tex")
        arows = []
        for k, g in self.cfg.items():
            if not g.get("factor") or not g.get("baseline"):
                continue
            base = self.group(g["baseline"])
            sv = c.config_summary(g, "val400")
            sb = c.config_summary(base, "val400") if base else {"AP": None, "n_runs": 0, "AP_std": None}
            arows.append([tex(g["factor"]), tex(g.get("setting", "")), tex(base.get("label", "") if base else ""),
                          str(sv["n_runs"]) if sv["n_runs"] else TBD, pm(sv["AP"], sv["AP_std"]),
                          v(signed(sv["AP"] - sb["AP"])) if sv["AP"] is not None and sb["AP"] is not None else TBD])
        if arows:
            self._tabular(p, ["One factor changed at a time relative to the stated default. Val400 only."], "@{}lllccc@{}",
                          ["Factor & Setting & Applied to & Seeds & Val400 AP & $\\Delta$\\\\"], arows)
            written.append(p)

        # sensitivity
        p = os.path.join(gen_dir, "tab_sensitivity.tex")
        g = self.group("rel_fidip_w48")
        rid = g["run_ids"][0] if g and g["run_ids"] else None
        best = self.group(role="final")
        brid = best["run_ids"][0] if best and best["run_ids"] else None

        def ap_of(r, split, proto="paper"):
            if not r:
                return TBD
            if best and r == brid:                       # best config: mean over its seeds
                s_ = c.config_summary(best, split, proto)
                return v(num1(s_["AP"])) if s_["n_runs"] else TBD
            val, _ = self.metric(r, split, "AP", proto)
            return v(pct1(val))
        sv = self.values
        srows = [
            ["Matching protocol (D10, D13)", "\\texttt{paper}: ids as shipped (headline)", "Test100", ap_of(rid, "test100"),
             ap_of(brid, "test100") if brid else "---"],
            ["", "\\texttt{fixed}: ids shifted off 0", "Test100", ap_of(rid, "test100", "fixed"),
             ap_of(brid, "test100", "fixed") if brid else "---"],
            "\\midrule",
            ["Train/test overlap (D16)", "Test100 as shipped (100)", "Test100", ap_of(rid, "test100"),
             ap_of(brid, "test100") if brid else "---"],
            ["", "Test100-clean (95; copies of training images removed)", "Test100-clean", ap_of(rid, "test100_clean"),
             ap_of(brid, "test100_clean") if brid else "---"],
            "\\midrule",
            ["Test images (D17)", "\\texttt{SyRIP.zip} (ours)", "Test100", ap_of(rid, "test100"),
             ap_of(brid, "test100") if brid else "---"],
            ["", "\\texttt{syrip\\_for\\_train.zip} (test11 = copy of test24)", "Test100", ap_of(rid, "validate_infant"), "---"],
            "\\midrule",
            ["GT boxes of the same 100 images", "validate100 (as shipped)", "Test100", v(sv.get("sens.box.v100.ap", (None,))[0]), "---"],
            ["", "validate500 ($\\text{area}=w h$)", "Test100", v(sv.get("sens.box.v500.ap", (None,))[0]), "---"],
            "\\midrule",
            ["OKS area on Val400 (D8)", "shipped constant 2443298.48", "Val400", v(sv.get("sens.area.const.ap", (None,))[0]), "---"],
            ["", "$\\text{area} = w h$ (ours)", "Val400", ap_of(rid, "val400"), ap_of(brid, "val400") if brid else "---"],
            "\\midrule",
            ["Metric code", "\\texttt{eval\\_coco.py} (pycocotools 2.0.8)", "Test100", ap_of(rid, "test100"),
             ap_of(brid, "test100") if brid else "---"],
            ["", "MMPose \\texttt{CocoMetric} (xtcocotools)", "Test100", self._xtcoco_ap(rid), "---"],
        ]
        self._tabular(p, ["Evaluation-condition sensitivity: same predictions, one evaluation choice changed per row pair."],
                      "@{}lllcc@{}", ["Condition & Variant & Split & Released FiDIP & Best config.\\\\"], srows)
        written.append(p)

        # per-keypoint groups (released vs best)
        p = os.path.join(gen_dir, "tab_perkeypoint.tex")
        groups = ["Head (nose, eyes, ears)", "Shoulders", "Elbows", "Wrists", "Hips", "Knees", "Ankles", "All"]
        rel_g = self._kp_groups(rid, "test100") if rid else {}
        best_g = self._kp_groups(brid, "test100") if brid else {}
        prow = []
        for gname in groups:
            if gname == "All":
                prow.append("\\midrule")
            a, b = rel_g.get(gname, {}), best_g.get(gname, {})
            prow.append([gname] + [v(num1(a.get(k))) for k in ("median_err_pct", "pck@0.05", "pck@0.1")]
                        + [v(num1(b.get(k))) for k in ("median_err_pct", "pck@0.05", "pck@0.1")])
        self._tabular(p, ["Per-keypoint error normalised by sqrt(GT bbox w*h), labelled joints, L/R pooled (Test100)."],
                      "@{}lccc ccc@{}",
                      ["& \\multicolumn{3}{c}{Released FiDIP (R001)} & \\multicolumn{3}{c}{Best configuration}\\\\",
                       "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
                       "Keypoint & med.\\ err.\\ (\\%) & PCK@.05 & PCK@.1 & med.\\ err.\\ (\\%) & PCK@.05 & PCK@.1\\\\"], prow)
        written.append(p)

        # failures
        p = os.path.join(gen_dir, "tab_failures.tex")
        fr = self._failures(rid, brid)
        self._tabular(p, ["Automatic failure taxonomy (eval/taxonomy.py) on Test100."], "@{}llcccc@{}",
                      ["& & \\multicolumn{2}{c}{\\# images} & \\multicolumn{2}{c}{mean OKS}\\\\",
                       "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
                       "Failure mode & Rule & Released & Best & Released & Best\\\\"], fr)
        written.append(p)
        return written

    def _xtcoco_ap(self, rid):
        for p in find_tables(self.tdir, f"sensitivity_extra_{rid}.json"):
            d = json.load(open(p))
            if d.get("test100_AP_xtcocotools") is not None:
                return pct1(d["test100_AP_xtcocotools"])
        return TBD

    def _kp_groups(self, rid, split):
        out = {}
        for p in find_tables(self.tdir, f"per_keypoint_groups_{split}.csv"):
            for r in read_csv(p):
                if rid in (r.get("run_ids") or "").split(";"):
                    out[r["group"]] = {k: fnum(r[k]) for k in ("median_err_pct", "pck@0.05", "pck@0.1")}
            if out:
                break
        return out

    def _failures(self, rid, brid):
        rows_by = {}
        for p in find_tables(self.tdir, "per_image_taxonomy_test100.csv"):
            for r in read_csv(p):
                rows_by.setdefault(p, []).append(r)
        def pick(run):
            """per-image rows with oks/swap/mirror columns for the config holding `run`."""
            if not run:
                return None, None
            key = next((g["key"] for g in self.cfg.values() if run in g["run_ids"]), None)
            for p, rows in rows_by.items():
                if rows and f"oks_{key}" in rows[0]:
                    return rows, key
            return None, None
        modes = [("Left/right swap", "$\\geq$1 L/R pair swapped (OKS gain $>$0.02, error halved)",
                  lambda r, k: bool(r.get(f"swaps_{k}"))),
                 ("Whole-body mirror", "mirroring all L/R labels raises OKS",
                  lambda r, k: (fnum(r.get(f"mirror_gain_{k}")) or 0) > 0),
                 ("Heavy occlusion", "$\\geq$2 GT joints with $v{=}1$", lambda r, k: (fnum(r.get("n_v1")) or 0) >= 2),
                 ("Inverted torso", "$|$torso angle$|>135^\\circ$", lambda r, k: r.get("inplane") == "inverted"),
                 ("Low OKS overall", "OKS $<$ 0.85", lambda r, k: (fnum(r.get(f"oks_{k}")) or 1) < 0.85)]
        rel, kr = pick(rid)
        bst, kb = pick(brid)
        out = []
        for name, rule, fn in modes:
            cells = []
            for rows, k in ((rel, kr), (bst, kb)):
                if rows is None:
                    cells.append((TBD, TBD))
                    continue
                sel = [r for r in rows if fn(r, k)]
                mo = sum(fnum(r[f"oks_{k}"]) for r in sel) / len(sel) if sel else None
                cells.append((str(len(sel)), f"{mo:.3f}" if mo is not None else "---"))
            out.append([name, rule, cells[0][0], cells[1][0], cells[0][1], cells[1][1]])
        return out

    # ---- figures for the report
    FIGURES = OrderedDict([
        ("oks_thresholds", ["compare_test100/ap_vs_oks_threshold_test100", "R001_fidip-w48_released/test100/ap_vs_oks_threshold_test100"]),
        ("per_keypoint", ["compare_test100/per_keypoint_error_test100", "R001_fidip-w48_released/test100/per_keypoint_error_test100"]),
        ("qual_success", ["R001_fidip-w48_released/test100/gallery_R001_fidip-w48_released_test100"]),
        ("qual_failure", ["R001_fidip-w48_released/test100/failures_R001_fidip-w48_released_test100"]),
        ("val_vs_test", ["val_vs_test"]),
        ("curves_fidip", ["curves/training_curves_fidip_retrain"]),
    ])

    def copy_figures(self, fig_root, dest):
        os.makedirs(dest, exist_ok=True)
        done = {}
        for name, cands in self.FIGURES.items():
            for cand in cands:
                src = os.path.join(fig_root, cand)
                if os.path.exists(src + ".pdf") and os.path.exists(src + ".png"):
                    for ext in ("pdf", "png"):
                        shutil.copyfile(f"{src}.{ext}", os.path.join(dest, f"{name}.{ext}"))
                    done[name] = cand
                    break
        return done

    def val_vs_test_figure(self, out_base):
        pts = getattr(self, "vt_points", [])
        if len(pts) < 2:
            return None
        import sys
        sys.path.insert(0, os.path.join(self.repo, "eval"))
        import plot_style as ps
        ps.apply()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(ps.COL_W, ps.COL_W * 0.85))
        lo = min(min(p[1], p[2]) for p in pts) - 2
        ax.plot([lo, 100], [lo, 100], color=ps.AXIS, lw=0.8)
        for rid, va, te in pts:
            g = next((g for g in self.cfg.values() if rid in g["run_ids"]), {})
            ax.scatter([va], [te], s=22, color=ps.color(g.get("slot")), edgecolor="white", lw=0.5, zorder=3)
            ax.annotate(g.get("label", rid), (va, te), fontsize=5.5, xytext=(3, -8), textcoords="offset points",
                        color=ps.INK2)
        ax.set_xlabel("Val400 AP (%)")
        ax.set_ylabel("Test100 AP (%, paper protocol)")
        ax.set_title("selection split vs reporting split", loc="left", fontsize=7.5)
        fig.tight_layout()
        return ps.save(fig, out_base)


def run(c, tables, repo, tables_dir, gen_dir, fig_root, fig_dest):
    ri = ReportInterface(c, tables, repo, tables_dir)
    ri.fill_scalars()
    n_set, n_keys = ri.write_numbers(os.path.join(gen_dir, "numbers.tex"))
    ri.write_provenance(os.path.join(tables_dir, "provenance.csv"))
    tabs = ri.write_tables(gen_dir)
    vt = ri.val_vs_test_figure(os.path.join(fig_root, "val_vs_test"))
    figs = ri.copy_figures(fig_root, fig_dest)
    return {"numbers_set": n_set, "numbers_keys": n_keys, "tables": [os.path.basename(t) for t in tabs],
            "figures": figs, "val_vs_test": bool(vt)}
