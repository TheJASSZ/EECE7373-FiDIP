#!/usr/bin/env python3
"""Fill report keys that scripts/collect_results.py does not (yet) emit, and set the pre-registered
H1 verdict sentence. Every value is computed from eval/eval_coco.py or eval/bootstrap_ci.py outputs.
Nothing is typed by hand.

Writes a marked, idempotent block into report/generated/manual_numbers.tex and replaces its
\\finaloutcome line. Also writes one provenance row per key to results/tables/provenance_final_keys.csv (overwritten each run).

Inputs (local mirror; run `bash scripts/pull_analysis.sh` first):
  results/analysis_eval/<RUN_ID>/metrics_<split>[_best].json      (eval_coco outputs)
  results/cluster_runs/<RUN_ID>/metrics_<split>[_best].json       (fallback)
  results/tables/bootstrap_final_C1_C4_test100.json               (contrasts C1-C4)
Decision rule H1 (results/PREREGISTRATION.md, D13/D20): the arm-B 5-seed mean Test100 AP (paper
protocol) is > 93.6, AND C1's paired-bootstrap CI excludes 0 under both protocols after Holm correction.
"""
import csv, glob, json, os, re, statistics as st, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(REPO, "report", "generated", "manual_numbers.tex")
PROV = os.path.join(REPO, "results", "tables", "provenance_final_keys.csv")   # overwritten each run (idempotent)
PUBLISHED = 0.936

def find_metrics(run_prefix, split, tag=""):
    for root in ("results/analysis_eval", "results/cluster_runs"):
        hits = sorted(glob.glob(os.path.join(REPO, root, f"{run_prefix}*", f"metrics_{split}{tag}.json")))
        if hits:
            return hits[0]
    return None

def ap(path, proto="metrics"):
    return json.load(open(path))[proto]["AP"]

def pct(x):
    return f"{100*x:.1f}"

def signed(x):
    v = round(100 * x, 1)
    return f"$+{v:.1f}$" if v >= 0 else f"$-{abs(v):.1f}$"

def mean_sd(runs, split, tag=""):
    files = [find_metrics(r, split, tag) for r in runs]
    if not all(files):
        return None, None, [f for f in files if f]
    vals = [ap(f) for f in files]
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), files

keys, prov = {}, []

def put(k, v, src, runs):
    keys[k] = v
    prov.append([k, v, " ".join(runs), src, "paper", src.split("metrics_")[-1].split(".json")[0] if "metrics_" in src else "test100"])

# plain fine-tune control R021-R023 (repro-3): best-Val400 checkpoints
ft = ["R021", "R022", "R023"]
for split, key in (("test100", "rt.ftnp.test.ap"), ("val400", "rt.ftnp.val.ap")):
    m, s, files = mean_sd(ft, split, "_best")
    if m is not None:
        put(key + ".mean", pct(m), ";".join(os.path.relpath(f, REPO) for f in files), ft)
        put(key + ".sd", pct(s), ";".join(os.path.relpath(f, REPO) for f in files), ft)

# arm A (HRNet-W48 control, fresh seeds R131-R135) and arm B (ViTPose-H 256, R136-R140), Test100
armA = [f"R{i}" for i in range(131, 136)]
armB = [f"R{i}" for i in range(136, 141)]
mA, sA, fA = mean_sd(armA, "test100")
if mA is not None:
    put("armA.test.ap.mean", pct(mA), ";".join(os.path.relpath(f, REPO) for f in fA), armA)
    put("armA.test.ap.sd", pct(sA), ";".join(os.path.relpath(f, REPO) for f in fA), armA)
mB, sB, fB = mean_sd(armB, "test100")

# contrasts C2, C3 (paired bootstrap, paper protocol)
boot = os.path.join(REPO, "results", "tables", "bootstrap_final_C1_C4_test100.json")
c1 = None
if os.path.exists(boot):
    bd = json.load(open(boot))
    for c in bd.get("contrasts", []):
        tag = c["name"].split(":")[0].strip().lower()           # "c1".."c4"
        d = c["delta"]["AP"]
        if tag in ("c2", "c3"):
            put(f"{tag}.delta", signed(d["observed"]), os.path.relpath(boot, REPO), [tag.upper()])
            put(f"{tag}.ci.lo", signed(d["ci95"][0]), os.path.relpath(boot, REPO), [tag.upper()])
            put(f"{tag}.ci.hi", signed(d["ci95"][1]), os.path.relpath(boot, REPO), [tag.upper()])
        if tag == "c1":
            c1 = c

# H1 verdict sentence (pre-registered rule), computed, not typed
if mB is not None and c1 is not None:
    d1 = c1["delta"]["AP"]
    both = bool(c1.get("ci_excludes_zero_both_protocols"))
    holm = bool(c1.get("reject_h0_holm"))
    robust = c1.get("robust_claim")
    cond_mean = mB > PUBLISHED
    cond_ci = (bool(robust) if robust is not None else (both and holm)) and d1["observed"] > 0
    verdict = "supported" if (cond_mean and cond_ci) else "not supported"
    why = []
    why.append(f"the arm-B 5-seed mean Test100 AP is {pct(mB)}$\\pm${pct(sB)} "
               f"({'above' if cond_mean else 'not above'} the published 93.6)")
    why.append(f"C1 (arm B vs released FiDIP) gives $\\Delta$AP {signed(d1['observed'])} "
               f"[{signed(d1['ci95'][0])}, {signed(d1['ci95'][1])}], and its CI "
               f"{'excludes' if cond_ci else 'does not exclude'} 0 under both protocols after Holm correction")
    sentence = (f"\\textbf{{Pre-registered hypothesis H1 is {verdict}:}} " + "; ".join(why) + ".")
    prov.append(["finaloutcome", verdict, " ".join(armB) + " R001", os.path.relpath(boot, REPO), "paper+fixed", "test100"])
else:
    sentence = None

# write block into manual_numbers.tex
txt = open(MAN).read()
begin, end = "% BEGIN fill_final_keys (generated by scripts/fill_final_keys.py)", "% END fill_final_keys"
block = [begin] + [f"\\setres{{{k}}}{{{v}}}" for k, v in keys.items()] + [end]
if begin in txt:
    txt = re.sub(re.escape(begin) + r".*?" + re.escape(end), "\n".join(block).replace("\\", "\\\\"), txt, flags=re.S)
else:
    txt = txt.replace("\\newcommand{\\finaloutcome}", "\n".join(block) + "\n\\newcommand{\\finaloutcome}", 1)
if sentence:
    txt = re.sub(r"\\newcommand\{\\finaloutcome\}\{.*\}\s*$", lambda m: "\\newcommand{\\finaloutcome}{" + sentence + "}\n", txt, flags=re.M)
open(MAN, "w").write(txt)

with open(PROV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["key", "value", "run_ids", "source_file", "protocol", "split"])
    for row in prov:
        w.writerow(row)
print(f"set {len(keys)} keys:", ", ".join(f"{k}={v}" for k, v in keys.items()))
print("finaloutcome:", sentence or "(not yet: arm-B metrics or C1 bootstrap missing)")
