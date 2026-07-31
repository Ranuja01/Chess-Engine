# -*- coding: utf-8 -*-
"""Pool the DEF=5-active collapse decision_fens (night_def5 + _s1 + _s2) into the next-round corpus. These are
real SF@2400 failures WITH KS_SAFE_CHECK_DEF=5 active, so they are the layer AFTER the defensive-KS fix -- the
next class to diagnose (use SF18-search triage, not SF11-static which is contaminated for attacks)."""
import os, csv, glob
THIS = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(THIS, "..", "selfplay", "games")
tags = ["night_def5", "night_def5_s1", "night_def5_s2"]
seen, rows = set(), []
for t in tags:
    cp = os.path.join(GAMES, t, "collapses.csv")
    if not os.path.exists(cp):
        print("MISSING", cp); continue
    n = 0
    for r in csv.DictReader(open(cp)):
        fen = (r.get("decision_fen") or "").strip()
        if fen and fen not in seen:
            seen.add(fen); rows.append(fen); n += 1
    print("%s: +%d unique decision_fens" % (t, n))
out = os.path.join(THIS, "ks_sets", "def5_remaining_collapses.txt")
with open(out, "w") as f:
    for fen in rows:
        f.write("def5_remaining\t%s\n" % fen)
print("wrote %d unique DEF=5-active collapse decision_fens -> %s" % (len(rows), out))
