#!/usr/bin/env python
"""Build analyze.py manifests for multi-model comparisons from configs.json + the analysis eval index.

  python scripts/analysis/build_manifests.py --out $R/analysis/manifests

Writes compare_val400.json (every configuration: Val400 is the selection set) and compare_test100.json
(only configurations allowed on Test100: released/retrain reproduction targets and pre-registered runs).

Colour follows the entity. A configuration keeps one categorical slot in every figure:
- configs.json "slot" if given;
- otherwise by role: released = 1, final/candidate = 2, control = 3, retrain = 4/5;
- any other role takes the next free slot.
Focus (coloured where all series meet) is the released, best final/candidate and control configs.
"""
import argparse
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import analysis_pipeline as apl  # noqa: E402

R = os.environ.get("R", "/scratch/purohitham.v/fidip")
ROLE_SLOT = {"released": 1, "final": 2, "candidate": 2, "control": 3, "retrain": 4}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=os.path.join(REPO, "results", "configs.json"))
    ap.add_argument("--eval-root", default=os.path.join(R, "analysis", "eval"))
    ap.add_argument("--out", default=os.path.join(R, "analysis", "manifests"))
    a = ap.parse_args(argv)
    cfg = json.load(open(a.configs))["configs"] if os.path.exists(a.configs) else {}
    reg, pre = apl.registry(), apl.prereg_ids()
    if not cfg:   # auto-group the registry: model + variant
        for rid, row in reg.items():
            k = re.sub(r"[^A-Za-z0-9]+", "_", f"{row.get('model', '')}_{row.get('variant', '')}").strip("_") or rid
            c = cfg.setdefault(k, {"label": k, "role": "released" if "released" in rid else
                                   ("retrain" if "_retrain" in rid else row.get("stream", "other")), "run_ids": []})
            c["run_ids"].append(rid)
    used, slots = set(), {}
    for k, c in cfg.items():
        if "slot" in c:                         # explicit, incl. 0 = grey context
            slots[k] = int(c["slot"])
            if slots[k]:
                used.add(slots[k])
            continue
        s = ROLE_SLOT.get(c.get("role"))
        if s and s not in used:
            slots[k] = s
            used.add(s)
    nxt = iter([s for s in range(1, 9) if s not in used])
    for k in cfg:
        if k not in slots:
            slots[k] = next(nxt, 0)
    os.makedirs(a.out, exist_ok=True)
    for split in ("val400", "test100"):
        models = []
        for k, c in cfg.items():
            if c.get("role") == "smoke":
                continue
            runs = []
            for rid in c.get("run_ids", []):
                pi = os.path.join(a.eval_root, rid, f"per_image_{split}.json")
                if not os.path.exists(pi):
                    continue
                if split == "test100" and not apl.test_allowed(rid, reg, pre)[0]:
                    continue
                runs.append({"run_id": rid, "per_image": pi, "metrics": pi.replace("per_image_", "metrics_")})
            if runs:
                models.append({"key": k, "label": c.get("label", k), "slot": slots[k], "runs": runs, "role": c.get("role")})
        order = {"released": 0, "retrain": 1, "control": 2, "candidate": 3, "final": 4, "ensemble": 5}
        models.sort(key=lambda m: (order.get(m.get("role"), 9), m["slot"] or 99))
        focus = [m["key"] for m in models if m.get("role") == "released"][:1]
        focus += [m["key"] for m in models if m.get("role") in ("final", "candidate")][:1]
        focus += [m["key"] for m in models if m.get("role") == "control"][:1]
        rel = next((m["key"] for m in models if m.get("role") == "released"), None)
        pairs = [[rel, m["key"]] for m in models if rel and m["key"] != rel and m.get("role") in
                 ("final", "candidate", "control", "retrain", "ensemble")]
        man = {"split": split, "gt": os.path.join(R, "data", "splits", f"{split}.json"), "models": models,
               "focus": focus or [m["key"] for m in models[:3]], "pairs": pairs,
               "published": {"label": "FiDIP (paper)", "AP": 0.936, "AP50": 0.985, "AP75": 0.985, "AR": 0.946}}
        if split == "test100":
            man["posture"] = os.path.join(R, "data", "raw", "syrip_for_train", "syrip", "annotations",
                                          "person_keypoints_validate_infant.json")
            vp = os.path.join(R, "analysis", "splits_derived", "README.json")
            if os.path.exists(vp):
                man["leak_ids"] = json.load(open(vp))["test100_clean"]["excluded_test_ids"]
        p = os.path.join(a.out, f"compare_{split}.json")
        json.dump(man, open(p, "w"), indent=1)
        print(f"{p}: {len(models)} configs, focus={man['focus']}")


if __name__ == "__main__":
    sys.exit(main())
