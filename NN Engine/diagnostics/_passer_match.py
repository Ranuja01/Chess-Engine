# -*- coding: utf-8 -*-
"""SF-free passer regression gate: run OUR engine over passers.csv and compare to the RECORDED sf_best /
sf_cp columns (no live Stockfish -> no interop hang). Honors PRESET/MAX_DEPTH + any engine env knobs, so
run once at baseline and once with a candidate config and diff the match%. Untracked dev probe.

Usage (from NN Engine/):
    PRESET=LIGHTNING USE_OPENING_BOOK=0 python diagnostics/_passer_match.py diagnostics/suites/passers.csv [N]
    PRESET=LIGHTNING USE_OPENING_BOOK=0 LMP_BASE=2 ... python diagnostics/_passer_match.py <csv> [N]
"""
import os, sys, csv
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from tactical_test import run_one

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
n = int(sys.argv[2]) if len(sys.argv) > 2 else len(rows)
if len(rows) > n:
    step = len(rows) / n
    rows = [rows[int(k * step)] for k in range(n)]

match = 0
tot = 0
abs_err = 0.0
for r in rows:
    fen = r["fen_start"]; sf_best = r.get("sf_best", ""); sf_cp = r.get("sf_cp", "")
    try:
        res = run_one(fen, set())
    except Exception:
        continue
    tot += 1
    if res["uci"] == sf_best:
        match += 1
    try:
        if res["eval"] is not None and sf_cp not in ("", None):
            abs_err += abs(res["eval"] / 10.0 - float(sf_cp))
    except Exception:
        pass
pct = 100.0 * match / tot if tot else 0.0
print(f"passer-match: {match}/{tot} ({pct:.1f}%)  mean|our_cp-sf_cp|={abs_err/tot if tot else 0:.0f}")
