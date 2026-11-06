"""Parse results/PREREGISTRATION.md into the set of RUN_ID prefixes ("R010") allowed on Test100.

- Markdown table rows only.
- Rows that say "no Test100" are skipped ("Val400 only picks the epoch" is not a ban).
- Ranges like "R010-R012" (or with an en dash) are expanded.
- Run IDs match on their 4-character prefix ("R010_fidip-w48_lambda5e-4_s1" -> "R010").
"""
import os
import re

RANGE = re.compile(r"\bR(\d{3})\s*[-–]\s*R(\d{3})\b")
SINGLE = re.compile(r"\bR(\d{3})(?=[_\s,;)`|]|$)")
FORBID = re.compile(r"\bno\s+test\s*100\b", re.I)   # explicit marker; "Val400 only picks the epoch" is NOT a ban


def allowed_prefixes(path):
    out = set()
    if not os.path.exists(path):
        return out
    for line in open(path):
        s = line.strip()
        if not s.startswith("|") or set(s) <= set("|-: ") or FORBID.search(s):
            continue
        for a, b in RANGE.findall(s):
            out.update(f"R{i:03d}" for i in range(int(a), int(b) + 1))
        out.update(f"R{m}" for m in SINGLE.findall(RANGE.sub(" ", s)))
    return out


def is_registered(run_id, prefixes):
    return run_id[:4] in prefixes or run_id in prefixes
