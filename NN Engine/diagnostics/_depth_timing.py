# -*- coding: utf-8 -*-
"""Time our FIXED-DEPTH search at the caller-set MAX_DEPTH over a stratified handful of corpus FENs, so the
low-depth-search tuner can be sized from data. PRESET=LONG_FORMAT + MAX_DEPTH=<d> must be set by the caller
(depth latches at engine init, so one process per depth). Reports median/mean/max wall time and mean nodes.

  for d in 4 5 6 7; do PRESET=LONG_FORMAT MAX_DEPTH=$d N=8 pyrun diagnostics/_depth_timing.py; done
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE); sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(ENGINE, "selfplay"))

import chess
from tactical_test import run_one

N = int(os.environ.get("N", "8"))
CORP = os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv")

rows = [r.get("fen", "").strip() for r in csv.DictReader(open(CORP)) if r.get("fen", "").strip()]
# Evenly-spaced sample across the file so phases (opening..endgame) are represented, not just the head.
step = max(1, len(rows) // N)
fens = rows[::step][:N]

run_one(fens[0], set())        # warm once (first search pays any one-time cost)
ts, ns = [], []
for f in fens:
    r = run_one(f, set())
    ts.append(r["time"]); ns.append(r.get("nodes") or 0)
ts_s = sorted(ts)
print("D=%s N=%d  median %.3fs  mean %.3fs  max %.3fs  mean_nodes %d" % (
    os.environ.get("MAX_DEPTH", "?"), len(ts), ts_s[len(ts_s)//2], sum(ts)/len(ts), ts_s[-1],
    sum(ns)//max(1, len(ns))), flush=True)
