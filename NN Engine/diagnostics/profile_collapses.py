# -*- coding: utf-8 -*-
"""Absolute per-CLASS collapse profile per config family, per-seed (so base vs gate is a fair per-seed
comparison, not a noisy vanish NET). Reads the classified dataset written by classify_collapses.py.
  pyrun diagnostics/profile_collapses.py [family1 family2 ...] [SEEDS=0,1]
Prints, per family: per-seed class counts + the per-seed AVERAGE per class over the chosen seeds.
"""
import os, sys, csv
from collections import defaultdict
THIS = os.path.dirname(os.path.abspath(__file__))
CSVP = os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv")
fams = [a for a in sys.argv[1:] if "=" not in a]
seeds = None
for a in sys.argv[1:]:
    if a.startswith("SEEDS="):
        seeds = set(int(x) for x in a.split("=", 1)[1].split(","))

rows = list(csv.DictReader(open(CSVP)))
# (family, seed) -> class -> count
cnt = defaultdict(lambda: defaultdict(int))
seen_seeds = defaultdict(set)
for r in rows:
    fam = r.get("family", "");
    try:
        sd = int(r.get("seed", ""))
    except (ValueError, TypeError):
        continue
    if fams and fam not in fams:
        continue
    if seeds is not None and sd not in seeds:
        continue
    cnt[(fam, sd)][r.get("ks_class", "?")] += 1
    seen_seeds[fam].add(sd)

CLASSES = ["ks_attack", "ks_and_material", "material", "positional"]
for fam in (fams or sorted(seen_seeds)):
    sds = sorted(seen_seeds[fam])
    if not sds:
        continue
    print("=== %s  (seeds %s) ===" % (fam, sds))
    tot = defaultdict(int)
    for sd in sds:
        line = "  s%d: " % sd + "  ".join("%s=%d" % (c, cnt[(fam, sd)][c]) for c in CLASSES)
        line += "   TOTAL=%d" % sum(cnt[(fam, sd)].values())
        print(line)
        for c in CLASSES:
            tot[c] += cnt[(fam, sd)][c]
    n = len(sds)
    print("  AVG/seed: " + "  ".join("%s=%.1f" % (c, tot[c] / n) for c in CLASSES)
          + "   TOTAL=%.1f" % (sum(tot.values()) / n))
