# -*- coding: utf-8 -*-
"""Scan the KS term over EVERY STS300 position under one KS config, dumping raw pre-floor units
(both kings) + the resulting KS term. Run once per config; a companion join (ks_band_join.py)
cross-tabs "KS newly fires" against "move regressed" to prove the STS regression lives in the
king-pressure band the fit's validation tiers never sampled.

  pyrun diagnostics/ks_band_scan.py off   # engine defaults
  pyrun diagnostics/ks_band_scan.py on    # KS_FLOOR=6 KS_SAFE_CHECK=8 KS_ATTACK_COUNT=2
Reads FENs from results/sts_results_ksoff.csv; writes results/ks_band_<mode>.csv.
"""
import os, sys, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mode = sys.argv[1] if len(sys.argv) > 1 else 'off'
if mode == 'on':
    os.environ['KS_FLOOR'] = '6'; os.environ['KS_SAFE_CHECK'] = '8'; os.environ['KS_ATTACK_COUNT'] = '2'

THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS, "results")
sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)

src = os.path.join(RES, "sts_results_ksoff.csv")
out = os.path.join(RES, f"ks_band_{mode}.csv")
rows = []
with open(src) as f:
    for r in csv.DictReader(f):
        b = chess.Board(r["fen"])
        bd = ai.ev_breakdown(b)
        uw = bd.get("det_ks_units_w", 0.0); ub = bd.get("det_ks_units_b", 0.0)
        ks = -bd.get("king_safety", 0.0) / 1000.0
        rows.append({"idx": r["idx"], "unitsW": uw, "unitsB": ub, "ks": round(ks, 3)})

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["idx", "unitsW", "unitsB", "ks"])
    w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows -> {out}")
