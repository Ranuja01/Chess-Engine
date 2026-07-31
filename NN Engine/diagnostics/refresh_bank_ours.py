# -*- coding: utf-8 -*-
"""Refresh ONLY the our_total / our_ks columns of position_bank.csv against the CURRENT build (e.g. after enabling
CAPG_PIN), leaving the expensive SF11 / SF18 / geometry labels untouched. Lets the KS/OvD fit re-run on the new
eval baseline cheaply. Run: pyrun diagnostics/refresh_bank_ours.py"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
# KEY=VAL argv -> environ BEFORE the ChessAI import (Config latches at init; pyrun forwards argv, not env),
# so the bank can be refreshed against a CANDIDATE baseline rather than only the compiled defaults.
for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import csv, chess
from ChessAI import ChessAI
ai = ChessAI(None, None, chess.Board(), True)
BANK = os.path.join(THIS, "ks_sets", "position_bank.csv")
rows = list(csv.DictReader(open(BANK))); cols = rows[0].keys()
changed = 0
for r in rows:
    try:
        bd = ai.ev_breakdown(chess.Board(r["fen"]))
        nt = round(-bd.get("total", 0.0) / 1000.0, 3); nk = round(-bd.get("king_safety", 0.0) / 1000.0, 3)
    except Exception:
        continue
    if abs(nt - float(r["our_total"])) >= 0.5: changed += 1
    r["our_total"] = nt; r["our_ks"] = nk
with open(BANK, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print("refreshed our_total/our_ks for %d rows; %d shifted >=0.5p (the pin-on delta)" % (len(rows), changed))
