# -*- coding: utf-8 -*-
"""Diff two STS result CSVs (baseline vs a candidate) to isolate the D10 POSITIONAL REGRESSIONS — positions the
candidate now scores lower than baseline. These are the collateral set: triangulate them vs SF to see how the
candidate mis-handles them. Writes the regression FENs to _sts_regress.fens for downstream SF triangulation.
Pure file processing. Run: bash <runner> pyrun diagnostics/_sts_regress.py <base_tag> <cand_tag>
"""
import os, sys, csv
THIS_DIR = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(THIS_DIR, 'results')
base_tag = sys.argv[1] if len(sys.argv) > 1 else 'stsbase'
cand_tag = sys.argv[2] if len(sys.argv) > 2 else 'stsv2'

def load(tag):
    d = {}
    p = os.path.join(RES, 'sts_results_%s.csv' % tag)
    for r in csv.DictReader(open(p)):
        sc = r['score']
        d[r['idx']] = (int(sc) if sc not in ('', None) else None, r['engine'], r['fen'], r['theme'], int(r['max']))
    return d

b, c = load(base_tag), load(cand_tag)
regs = []
for idx in b:
    if idx not in c: continue
    bs, bmv, fen, theme, mx = b[idx]
    cs, cmv, _, _, _ = c[idx]
    if bs is None or cs is None: continue
    if bs > cs:
        regs.append((bs - cs, theme, bs, mx, bmv, cs, cmv, fen))
regs.sort(reverse=True)

tot_drop = sum(r[0] for r in regs)
gains = sum(max(0, c[i][0] - b[i][0]) for i in b if i in c and b[i][0] is not None and c[i][0] is not None)
print("STS regressions %s -> %s:  %d positions dropped (total -%d pts); offsetting gains +%d" %
      (base_tag, cand_tag, len(regs), tot_drop, gains))
from collections import Counter
tc = Counter(r[1] for r in regs)
print("by theme:", "  ".join("%s=%d" % (k, v) for k, v in tc.most_common(8)))
print("\ntop drops (theme | base pts->move | cand pts->move | fen):")
with open(os.path.join(THIS_DIR, '_sts_regress.fens'), 'w') as f:
    f.write("fen_start\n")
    for drop, theme, bs, mx, bmv, cs, cmv, fen in regs:
        f.write(fen + "\n")
    for drop, theme, bs, mx, bmv, cs, cmv, fen in regs[:20]:
        print("  %-16s %d/%d->%s  %d->%s  | %s" % (theme, bs, mx, bmv, cs, cmv, fen))
